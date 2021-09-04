fairseq-preprocess --source-lang en --target-lang en \
    --trainpref bert/train.bpe --validpref bert/valid.bpe --testpref bert/test.bpe \
    --destdir bert/prepared/ --thresholdtgt 0 --thresholdsrc 0 \
    --workers 32

