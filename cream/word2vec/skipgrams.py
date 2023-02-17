import random

def skipgrams(
        sequence:list,
        vocabulary_size:int,
        window_size:int,
        shuffle:bool=True
    ):
    sg = []

    for t, cont in enumerate(sequence):
        for j in range(-1*window_size, window_size+1):
            if (j == 0): continue # pass (self, self)

            index = t + j
            if (index < 0 or index >= vocabulary_size): continue # skip negatives and overflowed index

            sg.append([cont, sequence[index]])

    if (shuffle): random.shuffle(sg)

    return sg