from openai import OpenAI
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_URL="http://localhost"
DEFAULT_PORT=11434
DEFAULT_API_KEY="ollama"
DEFAULT_MODEL="llama3:8b"

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

    Examples of forbidden output formats:
    "I hate when my code doesn’t work": this is wrong because the output is quoted
    """

def get_openai_client(url=DEFAULT_URL, port=DEFAULT_PORT, api_key=DEFAULT_API_KEY):
    return OpenAI(base_url=f"{url}:{port}/v1",api_key=api_key)

def generate_sentence(sentiment_type="positive", client=get_openai_client(), model="llama3:8b"):
    response = client.chat.completions.create(
        model=model,
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


def generate_sentence_task(sentiment_type, client, model):
    return generate_sentence(sentiment_type=sentiment_type, client=client, model=model)


def main():
    parser = argparse.ArgumentParser(description="An OpenAI client based sentence generator, designed to work primarily with Ollama")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=DEFAULT_API_KEY
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL
    )
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
        choices=("positive", "negative"),
        help="The sentiment to generate, either positive or negative. Defaults to positive.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Max workers to use in the ThreadPoolExecutor. Defaults to 2.",
    )
    parser.add_argument("--output", type=str, default="output.txt", help="The name of the output file.")

    args = parser.parse_args()
    client = get_openai_client(url=args.url,port=args.port,api_key=args.api_key)

    sentences = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(generate_sentence_task, args.sentiment, client, args.model)
            for _ in range(args.count)
        ]

        for future in tqdm(as_completed(futures), total=args.count):
            try:
                sentences.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    # for sentence in sentences:
    #     print(sentence)
    with open(args.output, 'w') as f:
        f.writelines('\n'.join(sentences))


if __name__ == "__main__":
    main()
