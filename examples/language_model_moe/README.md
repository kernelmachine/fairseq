# Mixture Models for Langage Modeling

## Preprocess data

```bash
TEXT=/net/nfs.corp/allennlp/suching/lm_data/mix/
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir data-bin/mix \
    --bpe sentencepiece \
    --nwordssrc 50257 \
    --workers 20
```

## Train a model

Then we can train a mixture of experts model using the `language_model_moe` task.
Use the `--method` flag to choose the MoE variant; we support hard mixtures with a learned or uniform prior (`--method hMoElp` and `hMoEup`, respectively) and soft mixures (`--method sMoElp` and `sMoEup`).
The model is trained with online responsibility assignment and shared parameterization.

The following command will train a `hMoEup` model with `10` experts:
```bash
fairseq-train 
--ddp-backend='no_c10d'  data-bin/mix/ \
--max-update 100000 \
--task language_model_moe \
--user-dir examples/language_model_moe/src \
--method hMoEup \
--arch transformer_lm \
--share-decoder-input-output-embed \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 \
--warmup-updates 4000 \
--lr 0.0007 \
--min-lr 1e-09 \
--dropout 0.1 \
--weight-decay 0.0 \
--criterion cross_entropy \
--max-tokens 3584 \
--num-experts 10 \
--mean-pool-gating-network \
--mean-pool-gating-network-encoder-dim 512 \
--mean-pool-gating-network-dropout 0.0   
```

## Generate

Once a model is trained, we can generate text from different experts using the `--gen-expert` option.
For example, to generate from expert 0:
```bash
fairseq-interactive data-bin/mix \
    --path checkpoints/checkpoint_best.pt \
    --beam 1 --sampling --sampling-topp 0.9 --remove-bpe \
    --task language_model_moe --user-dir examples/language_model_moe/src \
    --method hMoEup --mean-pool-gating-network \
    --mean-pool-gating-network-encoder-dim 512 \
    --mean-pool-gating-network-dropout 0.0 \  
    --num-experts 10 \
    --gen-expert 0
```

## Evaluate


```bash
fairseq-eval-lm data-bin/biomed/ \
--path checkpoints/checkpoint_best.pt \
--max-sentences 2 \
--tokens-per-sample 512 \
--context-window 400 \
--task language_model_moe  \
--user-dir ./examples/language_model_moe/src/ \
--num-experts 10 \
--mean-pool-gating-network \
--mean-pool-gating-network-encoder-dim 512 \
--mean-pool-gating-network-dropout 0.0
```
