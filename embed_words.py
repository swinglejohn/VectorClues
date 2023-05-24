import logging
import os
import pickle
import sys

import openai
from dotenv import load_dotenv
from fire import Fire
from more_itertools import chunked
from tenacity import (before_sleep_log, retry, retry_if_not_exception_type,
                      stop_after_attempt, wait_random_exponential)
from tqdm import tqdm

from utils import transform

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY ")
BATCH_SIZE = 100
TEXT_FILE = "data/corncob-58k-words.txt"

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
    with open(TEXT_FILE) as f:
        words = f.readlines()
    if n != "all":
        words = words[:n]
    words = [transform(word) for word in words]

    embeddings = {}
    for batch in tqdm(list(chunked(words, BATCH_SIZE))):
        batch_embeddings = get_embedding(batch)
        for word, embedding in zip(batch, batch_embeddings):
            embeddings[word] = embedding

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

from utils import DEFAULT_EMBEDDINGS, get_embeddings


def resave_as_numpy():
    embeddings = get_embeddings()
    with open(DEFAULT_EMBEDDINGS, "wb") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
      Fire()
