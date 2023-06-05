import pickle
from time import perf_counter_ns

import numpy as np
from colorama import Fore, Style

ALL_STYLES = ["", "spaces", "meaning"]
STYLE = ""

full_root = "data/58k_embeddings"
missing_root = "data/missing_embeddings"

# returns embedding file names given the style
def get_emb_file(style):
    full, missing = full_root, missing_root
    if style:
        full += f"_{style}"
        missing += f"_{style}"
    full += ".pkl"
    missing += ".pkl"
    return full, missing

DEFAULT_EMBEDDINGS, MISSING_EMBEDDINGS = get_emb_file(STYLE)

def transform(word, style=STYLE):
    word = word.strip()
    if style == "spaces":
        return " " + word
    elif style == "meaning":
        return f"The meaning of the word \"{word}\""
    elif style == "not-spelling":
        return f"The meaning and NOT the spelling of the word \"{word}\""
    elif style == "":
        return word
    else:
        raise NotImplementedError("untransform not implemented for style", style)

def untransform(word, style=STYLE):
    word = word.strip()
    if style == "spaces":
        return word
    # The spaces words aren't stored with their space
    #return word[1:]
    elif style == "meaning":
        return word[25:-1]
    elif style == "":
        return word
    else:
        raise NotImplementedError("untransform not implemented for style", style)

def load_embeddings(s=STYLE):
    default_embeddings, missing_embeddings = get_emb_file(s)
    print(f"Unpickling embeddings from {default_embeddings} and {missing_embeddings}")
    start = perf_counter_ns()
    with open(default_embeddings, "rb") as f:
        embeddings = pickle.load(f)
    if type(embeddings[transform("apple", s)]) == list:
        embeddings = {word: np.array(emb) for word, emb in embeddings.items()}
    del embeddings[transform("lochness", s)] # Loch Ness is two words and I added it to the missing words file
    # add in missing if not present (australia was missing from the 58k dictionary)
    if transform("australia", s) not in embeddings:
        with open(missing_embeddings, "rb") as f:
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
                toadd.append(input(f"{word} is not in the embeddings, re-enter: ").strip().lower())
                toremove.append(word)
        enemy =  [word for word in enemy if word not in toremove] + toadd

        #assert(len(set(friendly+enemy)) == len(friendly+enemy), "Duplicate words")
        print()

    print(f"({len(friendly + civilian + enemy)} words total):")
    print(f"Your team's words: {Fore.GREEN}{friendly}{Style.RESET_ALL}")
    print(f"Civilian words: {Fore.YELLOW}{civilian}{Style.RESET_ALL}")
    print(f"Enemy team and Assasin words: {Fore.RED}{enemy}{Style.RESET_ALL}")

    return friendly, civilian, enemy

