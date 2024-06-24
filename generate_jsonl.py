import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--positives", type=str, default="positives.txt")
parser.add_argument("--negatives", type=str, default="negatives.txt")
parser.add_argument("--output", type=str, default="data.jsonl")
args = parser.parse_args()

jsonl_format = '{"text": "%s", "label": %d}'
jsonl = []
with open(args.positives) as f:
    for line in f:
        jsonl.append(jsonl_format % (line.strip(), 1))

with open(args.negatives) as f:
    for line in f:
        jsonl.append(jsonl_format % (line.strip(), 0))

with open(args.output, 'w') as f:
    f.writelines('\n'.join(jsonl))
