from fire import Fire
import numpy as np
from tqdm import tqdm
from utils import transform, untransform, get_user_words, STYLE, get_embeddings
from colorama import Fore, Back, Style
import multiprocessing
from itertools import cycle
from more_itertools import chunked
from time import perf_counter_ns

# some colorama colors
# Fore.YELLOW, Fore.RED, Fore.GREEN, Fore.BLUE, Fore.CYAN, Fore.MAGENTA


# the freneemy distance is the difference between the last two distances (last is either enemy or a friendly deemed too far away)
def get_frenemey_diff(distances):
    if len(distances) > 1:
        frenemy_diff = distances[-1][1] - distances[-2][1]
    else:
        frenemy_diff = distances[-1][1]
    return frenemy_diff

def calculate(pid, friendly, enemy, friendly_embeddings, enemy_embeddings, emb_pairs):
    # lists of clues for 0, 1, 2,... words
    #print(f"Just entered Process {pid}, len emb_pairs: {len(emb_pairs)}")
    clues = [[] for _ in range(len(friendly) + 1)]

    for word, wembedding in emb_pairs:
        word = untransform(word, STYLE)
        skip = False
        if word in friendly + enemy:
            skip = True  # can't use a clue from the wordsets
        for target in friendly + enemy:
            if target in word or word in target:
                skip = True  # can't use subwords as clues or vice versa
        if skip:
            continue

        distances = []
        for target, tembedding in list(friendly_embeddings.items()) + list(enemy_embeddings.items()):
            distance = np.linalg.norm(tembedding - wembedding)
            distances.append((distance, target))
        distances.sort()

        # if the word is too close to an enemy, skip it
        enemy_cutoff = 0.58
        skip = False
        for i, (distance, target) in enumerate(distances):
            if distance > enemy_cutoff:
                break
            if target in enemy_embeddings:
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
            if distance > friendly_cutoff:
                break

    # mpq.put(clues)
    #print(f"inside process {pid}, len clues[2]: {len(clues[2])}")
    return clues

def run(default_words=4, embeddings=None, printn = 3, friendly = None, enemy = None):
    run_start = perf_counter_ns()
    print(f"{Fore.CYAN}Using embedding style: {Fore.YELLOW}{STYLE}{Style.RESET_ALL}")

    if not embeddings:
        embeddings = get_embeddings()

    if not friendly:
        friendly, enemy = get_user_words(embeddings, default=default_words)

    friendly_embeddings = {word:embeddings[transform(word, STYLE)] for word in friendly}
    enemy_embeddings    = {word:embeddings[transform(word, STYLE)] for word in enemy}
    emb_pairs = list(embeddings.items())
    print(f"Number of emb pairs: {len(emb_pairs)}")

    # speed up the calculation with multiprocessing
    num_processes = 12
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
        theargs = list(zip(pids, cycle([friendly]), cycle([enemy]), cycle([friendly_embeddings]), cycle([enemy_embeddings]), emb_sets))
        #print(f"len of theargs: {len(theargs)}")
        returned = pool.starmap(calculate, theargs)
        #print(f"len returned: {len(returned)}")
        for new_clues in returned:
            for i in range(len(clues)):
                clues[i].extend(new_clues[i])
    else:
        clues = calculate(0, friendly, enemy, friendly_embeddings, enemy_embeddings, emb_pairs)
    print(f"Finished calculating distances in {round((perf_counter_ns() - start) / 1e9, 2)} seconds")

    for tier in clues:
        # algorithms
        #tier.sort(key = lambda x: (x[2][0])) # Paul's
        #tier.sort(reverse=True, key=lambda x: x[1]) # sort for largest Frenemy distance
        tier.sort(key=lambda x: x[2][-2][1] if len(x[2])>1 else 1) # closest of last friendlies
        #tier.sort(key=lambda x: sum(y[1] for y in x[2][:-1]) if len(x[2])>1 else 1) # sum of all friendlies distances

    print()

    for i in range(1, len(clues)):
        if not clues[i]:
            break
        print(f"{Fore.LIGHTMAGENTA_EX}TOP {printn} CLUES FOR {i} WORDS:{Style.RESET_ALL}")
        for _, frenemy_diff, distances, word in clues[i][:printn]:
            print(f"\"{word}\" for {len(distances) - 1}")
            print(f"Frenemy Diff: {round(frenemy_diff, 4)}")
            print(" ".join([(Fore.GREEN if target in friendly else Fore.RED) + str(target) + Style.RESET_ALL + ", " + str(round(distance, 4)) for target, distance in distances]))
            print()

    print(f"Finished run in {round((perf_counter_ns() - run_start) / 1e9, 2)} seconds")

def get_distance(word1, word2):
    embeddings = get_embeddings()
    return np.linalg.norm(np.array(embeddings[transform(word1, STYLE)]) - np.array(embeddings[transform(word2, STYLE)]))

# get closest n words to input word
def get_closest(word, n=30):
    print(f"Transforming \"{word}\" to \"{transform(word, style=STYLE)}\"")
    word = transform(word, style=STYLE)

    embeddings = get_embeddings()
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


if __name__ == "__main__":
    Fire()
