time python spm_encode.py --model bert/sentencepiece.model --output_format=piece --inputs ../../data/prepared/bert/split/train.txt --outputs bert/train.bpe.en --max-len 128
time python spm_encode.py --model bert/sentencepiece.model --output_format=piece --inputs ../../data/prepared/bert/split/valid.txt  --outputs bert/valid.bpe.en --max-len 128
time python spm_encode.py --model bert/sentencepiece.model --output_format=piece --inputs ../../data/prepared/bert/split/test.txt --outputs bert/test.bpe.en --max-len 128
