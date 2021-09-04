fairseq-preprocess --source-lang de --target-lang en \
    --trainpref iwslt2016deen/train.bpe --validpref iwslt2016deen/valid.bpe --testpref iwslt2016deen/test.bpe \
    --destdir iwslt2016deen/prepared/ --thresholdtgt 0 --thresholdsrc 0 \
    --workers 32

