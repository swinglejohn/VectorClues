import numpy as np

from sort_words import run as sort_words
from utils import get_embeddings, get_user_words


def run():
    print("Press Ctrl-c to exit")
    embeddings = get_embeddings()
    embeddings = {word:np.array(emb) for word, emb in embeddings.items()}
    friendlies, civilian, enemies = get_user_words(embeddings=embeddings, default=0)
    while True:
        sort_words(default_words=None, friendly=friendlies, civilian=civilian, enemy=enemies)

        eliminated = input("Enter all words that have been eliminated: ").split()
        for word in eliminated:
            if word in friendlies:
                friendlies.remove(word)
            if word in enemies:
                enemies.remove(word)


if __name__ == "__main__":
    run()
