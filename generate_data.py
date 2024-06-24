from openai import OpenAI
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


def generate_prompt(sentiment_type):
    return f"""
    Generate strictly 1 (one) sentence that suggests a {sentiment_type} sentiment, in plain text, without any other complementary content.

    Examples of positive sentiment:
    I love programming
    Python is a great language
    I enjoy learning new things

    Examples of negative sentiment:
    I hate when my code doesn’t work
    I dislike when code is messy
    I hate syntax warnings

    Forbidden output formats:
    "I hate when my code doesn’t work": this is wrong because the output is quoted
    """


def generate_sentence(sentiment_type="positive"):
    response = client.chat.completions.create(
        model="llama3",
        temperature=2,
        messages=[
            {
                "role": "system",
                "content": "You are a data generator, responsible for generating data in plain text, as instructed. Provide strictly the requested output, nothing else.",
            },
            {"role": "user", "content": generate_prompt(sentiment_type)},
        ],
    )
    return response.choices[0].message.content


def generate_sentence_task(sentiment_type):
    return generate_sentence(sentiment_type)


def main():
    parser = argparse.ArgumentParser(description="A Llama3-8B based sentence generator")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="The number of sentences to generate. Defaults to 1.",
    )
    parser.add_argument(
        "--sentiment",
        type=str,
        default="positive",
        help="The sentiment to generate, either positive or negative. Defaults to positive.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Max workers to use in the ThreadPoolExecutor. Defaults to 2.",
    )

    args = parser.parse_args()
    sentences = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(generate_sentence_task, args.sentiment)
            for _ in range(args.count)
        ]

        for future in tqdm(as_completed(futures), total=args.count):
            try:
                sentences.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    for sentence in sentences:
        print(sentence)


if __name__ == "__main__":
    main()
