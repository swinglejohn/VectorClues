import logging
import os
import pickle
import sys

import numpy as np
import openai
from colorama import Fore, Style
from dotenv import load_dotenv
from fire import Fire
from more_itertools import chunked
from tenacity import (before_sleep_log, retry, retry_if_not_exception_type,
                      stop_after_attempt, wait_random_exponential)
from tqdm import tqdm

from utils import transform, STYLE

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = 100

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.INFO),
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
)
def get_embedding(texts, model="text-embedding-ada-002"):
   return [x['embedding'] for x in openai.Embedding.create(input = texts, model=model)['data']]

def run(n=5, embeddings_file_name="data/default_embeddings.pkl"):
    with open("data/corncob-58k-words.txt") as f:
        words = f.readlines()
    with open("data/missing-words.txt") as f:
        words.extend(f.readlines())
    if n != "all":
        words = words[:n]
    print(f"{Fore.CYAN}Using embedding style: {Fore.LIGHTGREEN_EX}{STYLE}{Style.RESET_ALL}")
    words = [transform(word) for word in words]

    embeddings = {}
    for batch in tqdm(list(chunked(words, BATCH_SIZE))):
        batch_embeddings = get_embedding(batch)
        for word, embedding in zip(batch, batch_embeddings):
            embeddings[word] = np.array(embedding)

    # save embeddings as a pickle file
    with open(embeddings_file_name, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Just embedded {len(embeddings)} words, stored in {embeddings_file_name}")

def print_first(n=10, embeddings_file_name="data/default_embeddings.pkl"):
    with open(embeddings_file_name, "rb") as f:
        embeddings = pickle.load(f)
    for word, embedding in list(embeddings.items())[:n]:
        print(f"\"{word}\"", len(embedding), embedding[:3])

def file_size():
    with open(TEXT_FILE) as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    print(len(words))


from utils import DEFAULT_EMBEDDINGS, MISSING_EMBEDDINGS, get_embeddings


# deprecated since we now store embeddings as numpy arrays
def resave_as_numpy():
    with open(MISSING_EMBEDDINGS, "rb") as f:
        embeddings = pickle.load(f)
    embeddings = {word: np.array(emb) for word, emb in embeddings.items()}
    print(embeddings)
    with open(MISSING_EMBEDDINGS, "wb") as f:
        pickle.dump(embeddings, f)


# count and view the words in codenames and not in the 58k dictionary
def missing_words():
    with open("data/codenames-words-duet.txt") as f:
        codenames_words = f.readlines()
    codenames_words = [word.strip().lower() for word in codenames_words]
    embeddings = get_embeddings()
    missing = []
    for word in codenames_words:
        if transform(word) not in embeddings:
            missing.append(word)
    print(f"There are {len(missing)} words missing:")
    print(missing)
    return missing


if __name__ == "__main__":
      Fire()
