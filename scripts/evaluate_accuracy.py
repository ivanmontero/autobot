import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("predictions")
parser.add_argument("classifier")
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.classifier)
model = model.cpu()
tok = AutoTokenizer.from_pretrained(args.classifier)
# print(model)
# values = []
correct = []
with open(args.predictions, "r") as fpred:
    for line in tqdm(fpred):
        outputs = model(**tok(line, return_tensors="pt"))
        correct.append((outputs[0][0,1] > outputs[0][0,0]).float().item())


print(sum(correct) / len(correct))
        # enc = tok.encode(line, return_tensors="pt")
    # lines = []
    # for line in fpred:
    #    lines.append(line)

    # enc = tok.batch_encode_plus(lines, return_tensors="pt")
    # pred.extend(model(enc)[:,0].numpy().tolist())
    # values.append(np.exp(-np.mean([float(v) for v in line.split(" ")])))
# print(np.mean(values))

