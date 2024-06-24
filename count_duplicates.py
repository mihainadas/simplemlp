import argparse

def count_duplicates(text):
    return (len(text), len(text) - len(set(text)))

def main():
    parser = argparse.ArgumentParser(description="Counts the duplicate lines in a file passed as argument.")
    parser.add_argument(
        "--input",
        type=str,
        required=True
    )
    args = parser.parse_args()
    with open(args.input) as f:
        input_len, input_duplicates = count_duplicates(f.readlines())
        print(f"{args.input} contains {input_len} lines, out of which {input_duplicates} are duplicate ({input_duplicates/input_len*100:.2f}%).")

if __name__ == "__main__":
    main()