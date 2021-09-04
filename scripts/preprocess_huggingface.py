import torch
import os
import subprocess
from transformers import AutoTokenizer
import json
from multiprocessing import Pool

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--trainpref", type=str)
    parser.add_argument("--validpref", type=str)
    parser.add_argument("--testpref", type=str)
    parser.add_argument("--destdir", type=str)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {}
    data["indices"] = tokenizer.get_vocab()
    special_tokens = {}
    special_tokens["bos_word"] = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
    special_tokens["pad_word"] = tokenizer.pad_token
    special_tokens["eos_word"] = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_word
    special_tokens["unk_word"] = tokenizer.unk_token
    special_tokens["mask_word"] = tokenizer.mask_token
    special_tokens["bos_index"] = data["indices"][special_tokens["bos_word"]]
    special_tokens["pad_index"] = data["indices"][special_tokens["pad_word"]]
    special_tokens["eos_index"] = data["indices"][special_tokens["eos_word"]]
    special_tokens["unk_index"] = data["indices"][special_tokens["unk_word"]]
    special_tokens["mask_index"] = data["indices"][special_tokens["mask_word"]]
    data["special_tokens"] = special_tokens

    print("Creating dictionary")
    os.makedirs(args.destdir, exist_ok=True)
    with open(os.path.join(args.destdir,'dictionary.json'), "w+", encoding="utf-8") as f:
        json.dump(data, f)
    
    def tokenize(s):
        tokens = [special_tokens["bos_word"]] + tokenizer.tokenize(s.rstrip()) + [special_tokens["eos_word"]]
        if len(tokens) > args.max_len:
            return None
        return ' '.join(tokens)

    print("Encoding train file")
    with open(args.trainpref + "." + args.lang, "r") as i, open(os.path.join(args.destdir, 'train.tok.' + args.lang), "w+") as o:
        lines = i.readlines()
        with Pool(args.workers) as p:
            tokenized = p.map(tokenize, lines)
        tokenized = [t for t in tokenized if t is not None]
        for l in tokenized:
            o.write(f"{l}\n")
    data_commands = f"--trainpref {os.path.join(args.destdir, 'train.tok')}"

    if args.validpref is not None:
        print("Encoding valid file")
        with open(args.validpref + "." + args.lang, "r") as i, open(os.path.join(args.destdir, 'valid.tok.' + args.lang), "w+") as o:
            lines = i.readlines()
            with Pool(args.workers) as p:
                tokenized = p.map(tokenize, lines)
            tokenized = [t for t in tokenized if t is not None]
            for l in tokenized:
                o.write(f"{l}\n")
        data_commands += f" --validpref {os.path.join(args.destdir, 'valid.tok')}"

    if args.testpref is not None:
        print("Encoding test file")
        with open(args.testpref + "." + args.lang, "r") as i, open(os.path.join(args.destdir, 'test.tok.' + args.lang), "w+") as o:
            lines = i.readlines()
            with Pool(args.workers) as p:
                tokenized = p.map(tokenize, lines)
            tokenized = [t for t in tokenized if t is not None]
            for l in tokenized:
                o.write(f"{l}\n")
        data_commands += f" --testpref {os.path.join(args.destdir, 'test.tok')}"

    print("Executing fairseq preprocessing")
    subprocess.call(
        f"""fairseq-preprocess \
            --source-lang {args.lang} --target-lang {args.lang} \
            {data_commands} \
            --arch autoencoder_roberta_base \
            --destdir {args.destdir} \
            --workers {args.workers} \
            --srcdict {os.path.join(args.destdir, "dictionary.json")} \
            --tgtdict {os.path.join(args.destdir, "dictionary.json")}""", shell=True)




# subprocess.call(f"mkdir -p {args.destdir}", shell=True)
# subprocess.call(f"wget -O {os.path.join(args.destdir,'encoder.json')} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json", shell=True)
# subprocess.call(f"wget -O {os.path.join(args.destdir,'vocab.bpe')} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe", shell=True)
# subprocess.call(f"wget -O {os.path.join(args.destdir,'roberta.dict.txt')} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt", shell=True)

# print("Encoding train file")
# subprocess.call(
#     f"""python -m examples.roberta.multiprocessing_bpe_encoder \
#         --encoder-json {os.path.join(args.destdir,'encoder.json')} \
#         --vocab-bpe {os.path.join(args.destdir,'vocab.bpe')} \
#         --inputs {args.trainpref + "." + args.lang} \
#         --outputs {os.path.join(args.destdir, 'train.bpe.' + args.lang)}  \
#         --keep-empty \
#         --workers {args.workers}""", shell=True)
# data_commands = f"--trainpref {os.path.join(args.destdir, 'train.bpe')}"

# if args.validpref is not None:
#     print("Encoding valid file")
#     subprocess.call(
#         f"""python -m examples.roberta.multiprocessing_bpe_encoder \
#             --encoder-json {os.path.join(args.destdir,'encoder.json')} \
#             --vocab-bpe {os.path.join(args.destdir,'vocab.bpe')} \
#             --inputs {args.validpref + "." + args.lang} \
#             --outputs {os.path.join(args.destdir, 'valid.bpe.' + args.lang)}  \
#             --keep-empty \
#             --workers {args.workers}""", shell=True)
#     data_commands += f" --validpref {os.path.join(args.destdir, 'valid.bpe')}"

# if args.testpref is not None:
#     print("Encoding test file")
#     subprocess.call(
#         f"""python -m examples.roberta.multiprocessing_bpe_encoder \
#             --encoder-json {os.path.join(args.destdir,'encoder.json')} \
#             --vocab-bpe {os.path.join(args.destdir,'vocab.bpe')} \
#             --inputs {args.testpref + "." + args.lang} \
#             --outputs {os.path.join(args.destdir, 'test.bpe.' + args.lang)}  \
#             --keep-empty \
#             --workers {args.workers}""", shell=True)
#     data_commands += f" --testpref {os.path.join(args.destdir, 'test.bpe')}"

# print("Executing fairseq preprocessing")
# subprocess.call(
#     f"""fairseq-preprocess \
#         --source-lang {args.lang} --target-lang {args.lang} \
#         {data_commands} \
#         --arch autoencoder_roberta_base \
#         --destdir {args.destdir} \
#         --workers {args.workers} \
#         --srcdict {os.path.join(args.destdir, "roberta.dict.txt")} \
#         --tgtdict {os.path.join(args.destdir, "roberta.dict.txt")}""", shell=True)


# # print("Downloading roberta")
# # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
# # print("Extracting roberta vocab and bpe")
# # roberta.task.dictionary.save(os.path.join(args.destdir, "roberta.dict.txt"))
# # bpe = roberta.bpe

# # print("Encoding train file")
# # os.makedirs(args.destdir, exist_ok=True)
# # with open(args.trainpref + "." + args.lang, "r") as sf, open(os.path.join(args.destdir, "train.bpe." + args.lang), "w+") as tf:
# #     for l in sf:
# #         tf.write("<s> " + bpe.encode(l.rstrip())  + " </s>\n")
# # data_commands = f"--trainpref {os.path.join(args.destdir, 'train.bpe')}"

# # if args.validpref is not None:
# #     print("Encoding valid file")
# #     with open(args.validpref + "." + args.lang, "r") as sf, open(os.path.join(args.destdir, "valid.bpe." + args.lang), "w+") as tf:
# #         for l in sf:
# #             tf.write("<s> " + bpe.encode(l.rstrip())  + " </s>\n")
# #     data_commands += f" --validpref {os.path.join(args.destdir, 'valid.bpe')}"

# # if args.testpref is not None:
# #     print("Encoding test file")
# #     with open(args.testpref + "." + args.lang, "r") as sf, open(os.path.join(args.destdir, "test.bpe." + args.lang), "w+") as tf:
# #         for l in sf:
# #             tf.write("<s> " + bpe.encode(l.rstrip())  + " </s>\n")
# #     data_commands += f" --testpref {os.path.join(args.destdir, 'test.bpe')}"

# # print("Executing fairseq preprocessing")
# # subprocess.call(
# # f"""fairseq-preprocess \
# #         --source-lang {args.lang} --target-lang {args.lang} \
# #         {data_commands} \
# #         --destdir {args.destdir} \
# #         --workers {args.workers} \
# #         --srcdict {os.path.join(args.destdir, "roberta.dict.txt")} --tgtdict {os.path.join(args.destdir, "roberta.dict.txt")}
# # """, shell=True)