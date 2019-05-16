'''
Measure differential privacy measure on a smoothed unigram LM
More specifically, it measures the diff between a smoothed unigram model when trained with and without one piece of text
'''

import sys
import collections
import numpy as np

def read_unigram(path, alpha):
    word2ind = {}
    ind2word = []
    ind2count = []
    with open(path) as f:
        for line in f:
            word, count = line.strip().split()
            word2ind[word] = len(word2ind)
            ind2word.append(word)
            ind2count.append(float(count)+alpha)

    return ind2word, word2ind, ind2count


def text2counts(input_filename):
    counts = collections.Counter()
    with open(input_filename, 'r') as f:
        for line in f:
            counts.update(line.strip().split())
    return counts


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: %s <input-filename> <vocab-filename> [smoothing-factor]" % sys.argv[0])
        sys.exit(1)

    input_filename = sys.argv[1]
    unigram_filename = sys.argv[2]
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    ind2word, word2ind, ind2count = read_unigram(unigram_filename, alpha)
    tot_count = sum(ind2count)

    input_word2count = text2counts(input_filename)
    tot_input_count = sum(input_word2count.values())

    s_pdtp = None

    for (word, input_count) in input_word2count.items():
        count  = ind2count[word2ind[word]]
        log_prob = np.log(count) - np.log(tot_count)
        log_prob_without = np.log(count - input_count) - np.log(tot_count - tot_input_count)
        diff_privacy = np.abs(log_prob - log_prob_without)
        if s_pdtp == None or s_pdtp < diff_privacy:
            s_pdtp = diff_privacy
            max_word_count = (word, count, input_count)

    print(s_pdtp)
