"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, LabelAccuracyEvaluator
from torch.utils.tensorboard import SummaryWriter
import gzip
import csv
import logging
from datetime import datetime
import torch.nn as nn
import sys
import os
from transformers import AutoTokenizer
import json

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="bert-base-uncased")
parser.add_argument("--save_path", type=str, default="savedir/sbert/test/")
parser.add_argument("--nhead_bottleneck", type=int, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--mode", type=str, default="train_and_evaluate", choices=["train_and_evaluate", "train", "evaluate"])
parser.add_argument("--valid_freq", type=int, default=1000)
args = parser.parse_args()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from barney import Barney

import torch
torch.manual_seed(args.seed)
import numpy as np
np.random.seed(args.seed)

base_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../")
data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../data/sbert/")

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
nli_dataset_path = os.path.join(base_folder, "data/sbert", 'AllNLI.tsv.gz')
sts_dataset_path = os.path.join(base_folder, "data/sbert", 'stsbenchmark.tsv.gz')

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


class BarneySentenceTransformerAdapter(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.barney = Barney.from_fairseq(path, 2, dropout=0.0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.barney.config.huggingface_model)

    def forward(self, features):
        return {'sentence_embedding': self.barney.forward_embedding(**features)}

    def get_sentence_embedding_dimension(self):
        return self.barney.config.fairseq_args["encoder_embed_dim"]

    def get_word_embedding_dimension(self):
        return self.barney.config.hidden_size

    def tokenize(self, text):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_sentence_features(self, tokens, pad_seq_length):
        pad_seq_length = min(pad_seq_length, 128, self.barney.config.max_position_embeddings-3) + 3 #Add space for special tokens

        if len(tokens) == 0 or isinstance(tokens[0], int):
            return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, padding='max_length', return_tensors='pt', truncation=True, prepend_batch_axis=True)
        else:
            return self.tokenizer.prepare_for_model(tokens[0], tokens[1], max_length=pad_seq_length, padding='max_length', return_tensors='pt', truncation='longest_first', prepend_batch_axis=True)

    def get_config_dict(self):
        return {}
        #return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        torch.save(self, os.path.join(output_path, "barney.pt"))
        # self.barney.save_pretrained(output_path)
        # self.tokenizer.save_pretrained(output_path)

        # with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
        #     json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        return torch.load(os.path.join(input_path, "barney.pt"))
        #Old classes used other config names than 'sentence_bert_config.json'
        # for config_name in ['sentence_bert_config.json']:
        #     sbert_config_path = os.path.join(input_path, config_name)
        #     if os.path.exists(sbert_config_path):
        #         break

        # with open(sbert_config_path) as fIn:
        #     config = json.load(fIn)
        # return Transformer(model_name_or_path=input_path, **config)

with SummaryWriter(log_dir=os.path.join(base_folder, args.save_path, "logs/")) as sw:

    if "train" in args.mode:
        #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
        model_name = args.model_path

        batch_size = args.batch_size
        model_save_path = os.path.join(base_folder, args.save_path, "model/")


        # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
        # word_embedding_model = models.Transformer(model_name)

        # Apply mean pooling to get one fixed sized sentence vector
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
        #                                pooling_mode_mean_tokens=True,
        #                                pooling_mode_cls_token=False,
        #                                pooling_mode_max_tokens=False)
        # pooling_model = CABPoolingWrapper(word_embedding_model.get_word_embedding_dimension(), args.nhead_bottleneck)

        barney = BarneySentenceTransformerAdapter(args.model_path)
        model = SentenceTransformer(modules=[barney])


        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        train_samples = []
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'train':
                    label_id = label2int[row['label']]
                    train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))


        train_dataset = SentencesDataset(train_samples, model=model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))


        #Read STSbenchmark dataset and use it as development set
        # logging.info("Read STSbenchmark dev dataset")
        # dev_samples = []
        # with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        #     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        #     for row in reader:
        #         if row['split'] == 'dev':
        #             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
        #             dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        logging.info("Read NLI dev dataset")
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        dev_samples = []
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'dev':
                    label_id = label2int[row['label']]
                    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))



        # dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=batch_size, name='sts-dev', main_similarity=SimilarityFunction.COSINE)
        
        dev_dataset = SentencesDataset(dev_samples, model=model)
        dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)
        dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, name="sts-dev", softmax_model=train_loss)


        # Configure the training
        num_epochs = args.epochs

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))


        # # Train the model
        # model.fit(train_objectives=[(train_dataloader, train_loss)],
        #         evaluator=evaluator,
        #         epochs=num_epochs,
        #         evaluation_steps=1000,
        #         warmup_steps=warmup_steps,
        #         output_path=model_save_path
        #         )

        # :param callback: Callback function that is invoked after each evaluation.
        #         It must accept the following three parameters in this order:
        #         `score`, `epoch`, `steps`
        def tensorboard_callback(score, epoch, steps):
            sw.add_scalar("Valid Score", score, steps)
            sw.add_scalar("Epoch", epoch, steps)

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=dev_evaluator,
                epochs=num_epochs,
                evaluation_steps=args.valid_freq,
                warmup_steps=warmup_steps,
                output_path=model_save_path,
                weight_decay=args.weight_decay,
                optimizer_params={'lr': args.lr, 'eps': 1e-6, 'correct_bias': False},
                callback=tensorboard_callback,
                )



    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    # model = SentenceTransformer(model_save_path)
    # 0_Transformer  1_CABPoolingWrapper
    if "evaluate" in args.mode:
        model_path = model_save_path if "train" in args.mode else args.model_path
        # os.path.join(base_folder, args.save_path, "model/")

        # word_embedding_model = models.Transformer.load(os.path.join(model_path, "0_Transformer"))
        # pooling_model = CABPoolingWrapper.load(os.path.join(model_path, "1_CABPoolingWrapper"))
        # BarneySentenceTransformerAdapter

        barney = BarneySentenceTransformerAdapter.load(os.path.join(model_path, "0_BarneySentenceTransformerAdapter"))
        model = SentenceTransformer(modules=[barney])

        # test_samples = []
        # with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        #     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        #     for row in reader:
        #         if row['split'] == 'test':
        #             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
        #             test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        logging.info("Read NLI test dataset")
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        test_samples = []
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'test':
                    label_id = label2int[row['label']]
                    test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))


        batch_size = args.batch_size
        # test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=args.batch_size, name='sts-test')
        
        test_dataset = SentencesDataset(test_samples, model=model)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
        test_evaluator = LabelAccuracyEvaluator(test_dataloader, name="sts-test", softmax_model=train_loss)
        
        test_evaluator(model, output_path=model_save_path)


