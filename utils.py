import pickle
from time import perf_counter_ns

import numpy as np
from colorama import Back, Fore, Style

STYLE = "meaning"
#STYLE = ""
DEFAULT_EMBEDDINGS = "data/58k_embeddings"
#DEFAULT_EMBEDDINGS = "data/10k_embeddings"
if STYLE:
    DEFAULT_EMBEDDINGS += f"_{STYLE}"
DEFAULT_EMBEDDINGS += ".pkl"

def transform(word, style):
    word = word.strip()
    if style == "spaces":
        return " " + word
    elif style == "meaning":
        return "The meaning of the word \"" + word + "\""
    else:
        return word

def untransform(word, style):
    word = word.strip()
    if style == "spaces":
        return word
        # The spaces words aren't stored with their space
        #return word[1:]
    elif style == "meaning":
        return word[25:-1]
    else:
        return word

def get_embeddings():
    print(f"Unpickling embeddings from {DEFAULT_EMBEDDINGS}")
    start = perf_counter_ns()
    with open(DEFAULT_EMBEDDINGS, "rb") as f:
        embeddings = pickle.load(f)
    if type(embeddings[transform("apple", STYLE)]) == list:
        embeddings = {word: np.array(emb) for word, emb in embeddings.items()}
    print(f"Unpickling took {round((perf_counter_ns() - start) / 1_000_000_000, 2)} s")
    return embeddings

def get_user_words(embeddings, default):
    friendly, civilian, enemy = [], [], []
    # "ninja" is not in the 10k word list
    if default==1:
        friendly, enemy = ["apple",  "fruit",  "angry",  "fight"], ["score",  "eat",  "ocean",  "play"]
    elif default==2:
        friendly, enemy = ["a", "aa"], ["aaa", "aaron"]
    elif default==3:
        friendly, enemy = ["a", "aa", "aaa", "aaron"], ["ab", "abandoned", "abc", "aberdeen"]
    elif default==4:
        friendly, enemy = ["aardvark", "aaron", "aback", "abacus"], ["abandon", "abandoned", "abacus", "abaft"]
    elif default==5:
        friendly, enemy = ["squirrel", "night", "heart", "contract", "mint", "sub", "horn", "war", "delta"], \
            ["barbecue", "anchor", "texas", "shoulder", "centaur", "violet", "sign", "stock", "embassy"]
    elif default==6:
        friendly, enemy = ["hawaii", "cliff", "pig", "milk", "snap", "ham", "spring", "microwave", "web"], \
            ["india", "bolt", "chalk", "farm", "butter", "maple", "china", "centaur", "second"]
    elif default==7:
        friendly, civilian, enemy = ["frog", "mustard", "string", "onion", "map", "war", "track", "bacon", "note"], \
            ["green", "revolution", "triangle", "paper", "pipe", "kiwi", "point", "silk", "rubber"], \
            ["cap", "tank", "mammoth", "sound", "crash", "foam", "superhero"]
    elif default==8:
        friendly = ["squirrel", "night", "heart", "contract", "mint", "sub", "horn", "war", "delta"]

    if friendly:
        print(f"Default words ({len(friendly+civilian+enemy)} total):")
        print(f"Your team's words: {Fore.GREEN}{friendly}{Style.RESET_ALL}")
        print(f"Civilian words: {Fore.YELLOW}{civilian}{Style.RESET_ALL}")
        print(f"Enemy team and Assasin words: {Fore.RED}{enemy}{Style.RESET_ALL}")
        return friendly, civilian, enemy

    else:
        print("Enter the CodeNames words, space separated")

        friendly = input("Your team's words: ").split()
        toadd, toremove = [], []
        for word in friendly:
            if transform(word, STYLE) not in embeddings:
                print(f"{word} not in embeddings")
                toadd.append(input("Re-enter: "))
                toremove.append(word)
        friendly =  [word for word in friendly if word not in toremove] + toadd

        civilian = input("Civilian words: ").split()
        toadd, toremove = [], []
        for word in civilian:
            if transform(word, STYLE) not in embeddings:
                print(f"{word} not in embeddings")
                toadd.append(input("Re-enter: "))
                toremove.append(word)
        civilian =  [word for word in civilian if word not in toremove] + toadd

        enemy = input("Enemy team and Assassin words: ").split()
        toadd, toremove = [], []
        for word in enemy:
            if transform(word, STYLE) not in embeddings:
                toadd.append(input(f"{word} is not in the embeddings, re-enter: "))
                toremove.append(word)
        enemy =  [word for word in enemy if word not in toremove] + toadd

        #assert(len(set(friendly+enemy)) == len(friendly+enemy), "Duplicate words")
        print()
        return friendly, civilian, enemy

