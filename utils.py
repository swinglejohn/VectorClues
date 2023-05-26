import pickle
from time import perf_counter_ns

import numpy as np
from colorama import Back, Fore, Style

STYLE = "meaning"
#STYLE = ""
DEFAULT_EMBEDDINGS = "data/58k_embeddings"
MISSING_EMBEDDINGS = "data/missing_embeddings"
if STYLE:
    DEFAULT_EMBEDDINGS += f"_{STYLE}"
    MISSING_EMBEDDINGS += f"_{STYLE}"
DEFAULT_EMBEDDINGS += ".pkl"
MISSING_EMBEDDINGS += ".pkl"

def transform(word):
    word = word.strip()
    if STYLE == "spaces":
        return " " + word
    elif STYLE == "meaning":
        return "The meaning of the word \"" + word + "\""
    else:
        return word

def untransform(word):
    word = word.strip()
    if STYLE == "spaces":
        return word
        # The spaces words aren't stored with their space
        #return word[1:]
    elif STYLE == "meaning":
        return word[25:-1]
    else:
        return word

def get_embeddings():
    print(f"Unpickling embeddings from {DEFAULT_EMBEDDINGS}")
    start = perf_counter_ns()
    with open(DEFAULT_EMBEDDINGS, "rb") as f:
        embeddings = pickle.load(f)
    if type(embeddings[transform("apple")]) == list:
        embeddings = {word: np.array(emb) for word, emb in embeddings.items()}
    del embeddings[transform("lochness")]
    # add in missing if not present (australia was missing from the 58k dictionary)
    if transform("australia") not in embeddings:
        with open(MISSING_EMBEDDINGS, "rb") as f:
            missing_embeddings = pickle.load(f)
        embeddings.update(missing_embeddings)
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
    elif default == 9: # the adversarial case containing all 12 words the 58k missed
        friendly, civilian, enemy = ["antarctica", "australia", "center", "himalayas", "hollywood", "ice cream", "loch ness", "ninja", "shakespeare"], \
            ["new york", "cable", "theater", "butter", "boxer", "capital", "spine"], \
            ["scuba diver", "rock", "university", "amazon", "mirror", "saddle", "frost", "brazil", "crane"]
    elif default == 10:
        friendly, civilian, enemy = ["slip", "hand", "flood", "centaur", "santa", "moscow", "pole", "seal", "potato"], \
            ["soap", "china", "pilot", "gymnast", "drop", "scarecrow", "ground"], \
            ["horseshoe", "bottle", "laundry", "whistle", "disk", "jeweler", "new york", "sleep", "russia"]
    else:
        print("Enter the CodeNames words, *comma* separated")

        friendly = list(w.strip().lower() for w in input("Your team's words: ").split(','))
        toadd, toremove = [], []
        for word in friendly:
            if transform(word) not in embeddings:
                print(f"{word} not in embeddings")
                toadd.append(input("Re-enter: "))
                toremove.append(word)
        friendly =  [word for word in friendly if word not in toremove] + toadd

        civilian = list(w.strip().lower() for w in input("Civilian words: ").split(','))
        toadd, toremove = [], []
        for word in civilian:
            if transform(word) not in embeddings:
                print(f"{word} not in embeddings")
                toadd.append(input("Re-enter: "))
                toremove.append(word)
        civilian =  [word for word in civilian if word not in toremove] + toadd

        enemy = list(w.strip().lower() for w in input("Enemy team and Assassin words: ").split(','))
        toadd, toremove = [], []
        for word in enemy:
            if transform(word) not in embeddings:
                toadd.append(input(f"{word} is not in the embeddings, re-enter: "))
                toremove.append(word)
        enemy =  [word for word in enemy if word not in toremove] + toadd

        #assert(len(set(friendly+enemy)) == len(friendly+enemy), "Duplicate words")
        print()

    print(f"({len(friendly + civilian + enemy)} words total):")
    print(f"Your team's words: {Fore.GREEN}{friendly}{Style.RESET_ALL}")
    print(f"Civilian words: {Fore.YELLOW}{civilian}{Style.RESET_ALL}")
    print(f"Enemy team and Assasin words: {Fore.RED}{enemy}{Style.RESET_ALL}")

    return friendly, civilian, enemy

