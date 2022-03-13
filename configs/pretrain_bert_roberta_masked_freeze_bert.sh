fairseq-train \
    data/bert_roberta/prepared/ \
    --arch autoencoder_roberta_base --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler linear --warmup-updates 1000 --max-update 10000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 6144 --seed 42 --num-workers 4 \
    --validate-interval 1 \
    --tensorboard-logdir savedir/fairseq/bert_roberta_masked_freeze_bert10_10k/logs/ \
    --save-dir savedir/fairseq/bert_roberta_masked_freeze_bert10_10k/model/ \
    --freeze-bert \
    --save-interval-updates 2500 \
    --keep-interval-updates 0 \
    --decoder-layers 1 \
    --dict-from-json \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --mask-words \
    --update-freq 4 \
    --log-interval 100
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 1}' \
#    --eval-bleu-remove-bpe \
#    --eval-bleu-detok moses \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
