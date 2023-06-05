import multiprocessing
from itertools import cycle
from time import perf_counter_ns

import numpy as np
from colorama import Fore, Style
from fire import Fire
from more_itertools import chunked
from tqdm import tqdm

from utils import (ALL_STYLES, STYLE, get_user_words, load_embeddings,
                   transform, untransform)


# the frenemy distance is the difference between the last two distances
# (the last distance is either enemy or a friendly deemed too far away)
def get_frenemey_diff(distances):
    if len(distances) > 1:
        frenemy_diff = distances[-1][1] - distances[-2][1]
    else:
        frenemy_diff = distances[-1][1]
    return frenemy_diff

# A helper function for generating clues
def calculate(pid, friendly, civilian, enemy, emb_pairs):
    # lists of clues for 0, 1, 2,... words
    clues = [[] for _ in range(len(friendly) + 1)]

    for word, wembedding in emb_pairs:
        word = untransform(word)
        skip = False
        if word in list(friendly) + list(enemy):
            skip = True  # can't use a clue from the wordsets
        for target in list(friendly) + list(enemy):
            if target in word or word in target:
                skip = True  # can't use subwords as clues or vice versa
        if skip:
            continue

        distances = []
        for target, tembedding in list(friendly.items()) + list(enemy.items()) + list(civilian.items()):
            distance = np.linalg.norm(tembedding - wembedding)
            distances.append((distance, target))
        distances.sort()

        # if the word is too close to an enemy, skip it
        enemy_cutoff = 0.58
        skip = False
        for i, (distance, target) in enumerate(distances):
            if distance > enemy_cutoff:
                break
            if target in enemy:
                skip = True
                break
        if skip:
            continue

        friendly_cutoff = 0.560
        new_distances = []
        for i, (distance, target) in enumerate(distances):
            new_distances.append((target, distance))
            frenemy_diff = get_frenemey_diff(new_distances)
            clues[i].append((len(new_distances) - 1, frenemy_diff, new_distances.copy(), word))
            if distance > friendly_cutoff or target not in friendly:
                break

    return clues

def run(default_words=7, embeddings=None, printn = 3, friendly = None, civilian = None, enemy = None):
    run_start = perf_counter_ns()
    print(f"{Fore.CYAN}Using embedding style: {Fore.LIGHTGREEN_EX}{STYLE}{Style.RESET_ALL}")

    if not embeddings:
        embeddings = load_embeddings()

    if not friendly:
        friendly, civilian, enemy = get_user_words(embeddings, default=default_words)

    friendly = {word:embeddings[transform(word)] for word in friendly}
    civilian = {word:embeddings[transform(word)] for word in civilian}
    enemy    = {word:embeddings[transform(word)] for word in enemy}
    emb_pairs = list(embeddings.items())

    # speed up the calculation with multiprocessing
    num_processes = multiprocessing.cpu_count()
    print(f"\nCalculating distances - {num_processes} additional processes")
    start = perf_counter_ns()
    if num_processes:
        pool = multiprocessing.Pool(processes=num_processes)
        clues = [[] for _ in range(len(friendly) + 1)]
        step = len(emb_pairs) // num_processes
        emb_sets = list(chunked(emb_pairs, step))
        if len(emb_sets) > num_processes:
            emb_sets[-2].extend(emb_sets[-1])
            emb_sets.pop()
        pids = list(range(1, num_processes+1))
        theargs = list(zip(pids, cycle([friendly]), cycle([civilian]), cycle([enemy]), emb_sets))
        #print(f"len of theargs: {len(theargs)}")
        returned = pool.starmap(calculate, theargs)
        #print(f"len returned: {len(returned)}")
        for new_clues in returned:
            for i in range(len(clues)):
                clues[i].extend(new_clues[i])
    else:
        clues = calculate(0, friendly, civilian, enemy, emb_pairs)
    print(f"Finished calculating distances in {round((perf_counter_ns() - start) / 1e9, 2)} seconds\n")

    for tier in clues:
        # algorithms for sorting potential clues
        #tier.sort(key = lambda x: (x[2][0])) # Paul's
        #tier.sort(reverse=True, key=lambda x: x[1]) # sort for largest Frenemy distance
        tier.sort(key=lambda x: x[2][-2][1] if len(x[2])>1 else 1) # closest of last friendlies
        #tier.sort(key=lambda x: sum(y[1] for y in x[2][:-1]) if len(x[2])>1 else 1) # sum of all friendlies distances

    def colorize(word):
        if word in friendly:
            return Fore.GREEN + word + Style.RESET_ALL
        elif word in enemy:
            return Fore.RED + word + Style.RESET_ALL
        elif word in civilian:
            return Fore.YELLOW + word + Style.RESET_ALL
        else:
            return word

    for i in range(1, len(clues)):
        if not clues[i]:
            break
        print(f"{Fore.LIGHTMAGENTA_EX}TOP {printn} CLUES FOR {i} WORDS,{Style.RESET_ALL} ({len(clues[i])} TOTAL)")
        for _, frenemy_diff, distances, word in clues[i][:printn]:
            print(f"\"{word}\" for {len(distances) - 1}")
            #print(f"Frenemy Diff: {round(frenemy_diff, 4)}")
            print(" ".join([colorize(target) + ", " + str(round(distance, 4)) for target, distance in distances]))
            print()

    print(f"Finished run in {round((perf_counter_ns() - run_start) / 1e9, 2)} seconds")

def get_distance(word1, word2):
    embeddings = load_embeddings()
    return np.linalg.norm(np.array(embeddings[transform(word1)]) - np.array(embeddings[transform(word2)]))

# get closest n words to input word
def get_closest(word, n=40):
    print(f"Transforming \"{word}\" to \"{transform(word)}\"")
    word = transform(word)

    embeddings = load_embeddings()
    distances = []
    for (target, tembedding) in tqdm(embeddings.items()):
        distances.append(("\"" + target + "\"", np.linalg.norm(np.array(tembedding) - np.array(embeddings[word]))))
    distances.sort(key=lambda x: x[1])

    print(f"The closest {n} words are:")
    for i in range(n):
        print(f"{i:<3} {str(distances[i][0]):<50} {round(distances[i][1], 5)}")

    x = 5
    print(f"\nThe furthest {x} words are:")
    for i in range(len(distances)-1-x, len(distances)):
        print(f"{i:<3} {str(distances[i][0]):<50} {round(distances[i][1], 5)}")

# compare closest n words for each different style side by side
# similar to get_closest but for each style all at once printed in columns
def compare_closest(word, n=40):
    distances = [[] for _ in range(len(ALL_STYLES))]
    for i, s in enumerate(ALL_STYLES):
        tword = transform(word, s)
        embeddings = load_embeddings(s)
        for (target, tembedding) in tqdm(embeddings.items()):
            distances[i].append(("\"" + target + "\"", np.linalg.norm(np.array(tembedding) - np.array(embeddings[tword]))))
        distances[i].sort(key=lambda x: x[1])

    # print styles as headers
    print("Styles to compare (beginning with just the plain word")
    for s in ALL_STYLES:
        print(f"{s:<48}", end=" | ")
    print()
    print(f"The closest {n} words to {word} are:")
    for words in zip(*(d[:n] for d in distances)):
        for c in words:
            print(f"{str(c[0]):<40} {round(c[1], 5):<7}", end=" | ")
        print()

    print()

if __name__ == "__main__":
    Fire()
