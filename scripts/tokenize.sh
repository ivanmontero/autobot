time python spm_encode.py --model iwslt2016deen/sentencepiece.model --output_format=piece --inputs ../../data/iwslt2016/deen/train.en --outputs iwslt2016deen/train.bpe.en
time python spm_encode.py --model iwslt2016deen/sentencepiece.model --output_format=piece --inputs ../../data/iwslt2016/deen/tst201314.en  --outputs iwslt2016deen/valid.bpe.en 
time python spm_encode.py --model iwslt2016deen/sentencepiece.model --output_format=piece --inputs ../../data/iwslt2016/deen/tst201516.en --outputs iwslt2016deen/test.bpe.en 
time python spm_encode.py --model iwslt2016deen/sentencepiece.model --output_format=piece --inputs ../../data/iwslt2016/deen/train.de --outputs iwslt2016deen/train.bpe.de
time python spm_encode.py --model iwslt2016deen/sentencepiece.model --output_format=piece --inputs ../../data/iwslt2016/deen/tst201314.de  --outputs iwslt2016deen/valid.bpe.de
time python spm_encode.py --model iwslt2016deen/sentencepiece.model --output_format=piece --inputs ../../data/iwslt2016/deen/tst201516.de --outputs iwslt2016deen/test.bpe.de
