# Introduction

This repository contains the code for the paper 
[Sentence Bottleneck Autoencoders from Transformer Language Models](https://arxiv.org/abs/2109.00055)
by Ivan Montero, Nikolaos Pappas, and Noah A. Smith published at EMNLP 2021.

This paper proposes an approach to learning sentence representations by applying an autoencoder on top of pretrained masked LMs.

Further documentation and code cleanup are under the works!

# Installation

Coming soon!

This code was tested under Ubuntu 18.04, Python 3.7, and PyTorch 1.6.

To install, run the setup file.
```
bash setup.sh
```

We use modified versions of the following repositories:
```
transformers==3.3.1
fairseq==0.9.0
```

# Usage

## Data Preprocessing

The `scripts/preprocess_huggingface.py` handles the process of preprocessing input files, which contain one sentence per line, using the tokenizer of a pretrained Huggingface tokenizer in a parallelized manner. For example, here is an example command that uses the roberta-base tokenizer.
```
python scripts/preprocess_huggingface.py --model roberta-base --lang en --trainpref <train_file> --validpref <valid_file> --testpref  <test_file> --workers 128 --max-len 128 --destdir data/yelp/processed/
```
For testing, one can use the WikiText-103 dataset specified [in the fairseq pretraining instructions](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.pretraining.md)

## Pretraining

An example pretraining script that takes preprocessed data and runs the autoencoding procedure is specified in `config/pretrain_bert_roberta_masked_freeze_bert.sh`. This file will need to be edited to work with the data processed in the data preprocessing stage

## Finetuning

An example finetuning script that takes a pretrained model and performs the STS task using the SBERT code in specified in `config/finetune_sbert_nli`.

TODO: Include glue and generation experiments

# Modified Files

Coming soon!