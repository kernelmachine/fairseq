# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq import metrics, utils
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask
from .logsumexp_moe import LogSumExpMoE
from .mean_pool_gating_network import MeanPoolGatingNetwork
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force="True")
done = mp.Event()
@register_task('language_modeling_moe')
class LanguageModelMoETask(LanguageModelingTask):
    """
    Language Modeling task for Mixture of Experts (MoE) models.

    See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
    (Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.

    Args:
        dictionary (~fairseq.data.Dictionary): dictionary for the source language
        output_dictionary (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        LanguageModelingTask.add_args(parser)
        parser.add_argument('--method', default='hMoEup',
                            choices=['sMoElp', 'sMoEup', 'hMoElp', 'hMoEup'])
        parser.add_argument('--num-experts', default=3, type=int, metavar='N',
                            help='number of experts')
        parser.add_argument('--mean-pool-gating-network', action='store_true',
                            help='use a simple mean-pooling gating network')
        parser.add_argument('--mean-pool-gating-network-dropout', type=float,
                            help='dropout for mean-pooling gating network')
        parser.add_argument('--mean-pool-gating-network-encoder-dim', type=int,
                            help='encoder output dim for mean-pooling gating network')
        parser.add_argument('--gen-expert', type=int, default=0,
                            help='which expert to use for generation')
        # fmt: on

    def __init__(self, args, dictionary, output_dictionary, targets=None):
        if args.method == 'sMoElp':
            # soft MoE with learned prior
            self.uniform_prior = False
            self.hard_selection = False
        elif args.method == 'sMoEup':
            # soft MoE with uniform prior
            self.uniform_prior = True
            self.hard_selection = False
        elif args.method == 'hMoElp':
            # hard MoE with learned prior
            self.uniform_prior = False
            self.hard_selection = True
        elif args.method == 'hMoEup':
            # hard MoE with uniform prior
            self.uniform_prior = True
            self.hard_selection = True

        # add indicator tokens for each expert
        for i in range(args.num_experts):
            # add to both dictionaries in case we're sharing embeddings
            dictionary.add_symbol('<expert_{}>'.format(i))
            output_dictionary.add_symbol('<expert_{}>'.format(i))

        super().__init__(args, dictionary, output_dictionary=output_dictionary, targets=targets)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not hasattr(model, 'gating_network'):
            if self.args.mean_pool_gating_network:
                if getattr(args, 'mean_pool_gating_network_encoder_dim', None):
                    encoder_dim = args.mean_pool_gating_network_encoder_dim
                elif getattr(args, 'encoder_embed_dim', None):
                    # assume that encoder_embed_dim is the encoder's output dimension
                    encoder_dim = args.encoder_embed_dim
                else:
                    raise ValueError('Must specify --mean-pool-gating-network-encoder-dim')

                if getattr(args, 'mean_pool_gating_network_dropout', None):
                    dropout = args.mean_pool_gating_network_dropout
                elif getattr(args, 'dropout', None):
                    dropout = args.dropout
                else:
                    raise ValueError('Must specify --mean-pool-gating-network-dropout')

                model.gating_network = MeanPoolGatingNetwork(
                    encoder_dim, args.num_experts, dropout,
                )
            else:
                raise ValueError(
                    'language_model_moe task with learned prior requires the model to '
                    'have a gating network; try using --mean-pool-gating-network'
                )
        return model

    def expert_index(self, i):
        return i + self.output_dictionary.index('<expert_0>')

    def _get_loss(self, sample, model, criterion):
        assert hasattr(criterion, 'compute_loss'), \
            'language_model_moe task requires the criterion to implement the compute_loss() method'

        bsz = sample['target'].size(0)
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']

        #### E-STEP
        with utils.eval(model):  # disable dropout
            with torch.no_grad():  # disable autograd
                net_output = model(
                    src_tokens=src_tokens,
                    src_lengths=src_lengths
                )
        # pass net output to gating network to compute expert probabilities
        expert_probs = model.gating_network(net_output)  
        # hard selection of experts
        expert_assignments = [self.expert_index(x) for x in expert_probs.argmax(dim=1)]
        # add expert assignments as BOS tokens
        src_tokens[:, 0] = torch.Tensor(expert_assignments).long()
        
        #### M-STEP
        net_output = model(
                src_tokens=src_tokens,
                src_lengths=src_lengths
        )
        loss, _ = criterion.compute_loss(model, net_output, sample, reduce=False)
        loss = loss.view(sample['target'].size(0), -1)
        loss = loss.sum(dim=1, keepdim=True)
    
        loss = loss.sum()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
            "expert_assignments": expert_probs.argmax(dim=1)
        }
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, expert=None):
        expert = expert or self.args.gen_expert

        with torch.no_grad():
            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                prefix_tokens[:,0] = self.expert_index(expert)
                prefix_tokens = prefix_tokens[:, 1:]
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                bos_token = self.expert_index(expert)
            )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        metrics.log_scalar(
            'posterior',
            sum(log['posterior'] for log in logging_outputs if 'posterior' in log)
        )
