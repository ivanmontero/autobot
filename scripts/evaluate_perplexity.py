import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("probabilities")
args = parser.parse_args()

values = []
with open(args.probabilities, "r") as fprob:
  for line in fprob:
    values.append(np.exp(np.mean([float(v) for v in line.split(" ")])))
print(np.mean(values))

