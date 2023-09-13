import numpy as np
from nltk.corpus import wordnet


def most_common_synset(word, poss=None):
    """
    Get the most common synset for a word. If a part of speech is specified, will search for the most common synset
    of that part of speech. If no synset is found for that part of speech, will return the most common synset of any
    part of speech.

    :param word:
    :param poss:
    :return:
    """
    synsets = wordnet.synsets(word)
    most_common = None
    most_common_n = -np.Inf

    if poss == None:
        for s in synsets:
            freq = 0
            for lemma in s.lemmas():
                freq += lemma.count()
            if freq > most_common_n:
                most_common = s
                most_common_n = freq

    else:
        if type(poss) in [str]:
            poss = [poss]

        i = 0
        while most_common is None and i < len(poss):
            pos = poss[i]
            for s in synsets:
                if s.pos() != pos:
                    continue
                freq = 0
                for lemma in s.lemmas():
                    freq += lemma.count()

                if freq > most_common_n:
                    most_common = s
                    most_common_n = freq
            i += 1
        if most_common is None:
            for s in synsets:
                freq = 0
                for lemma in s.lemmas():
                    freq += lemma.count()
                if freq > most_common_n:
                    most_common = s
                    most_common_n = freq

    return most_common
