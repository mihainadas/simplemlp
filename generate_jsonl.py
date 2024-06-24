import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--positives", type=str, default="positives.txt")
parser.add_argument("--negatives", type=str, default="negatives.txt")
args = parser.parse_args()

jsonl_format = '{"text": "%s", "label": %d}'
with open(args.positives) as f:
    for line in f:
        print(jsonl_format % (line.strip(), 1))

with open(args.negatives) as f:
    for line in f:
        print(jsonl_format % (line.strip(), 0))
