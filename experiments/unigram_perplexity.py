'''
Measures perplexity of a smoothed unigram model on note text
'''

import sys
import numpy as np

def read_unigram(path, alpha):
    ind2word = []
    ind2count = []
    word2ind = {}
    with open(path) as f:
        for line in f:
            word, count = line.strip().split()
            count = float(count) + alpha # lidstone smoothing
            ind2word.append(word)
            ind2count.append(count)
            word2ind[word] = len(word2ind)

    log_probabilities = np.log(np.asarray(ind2count)/sum(ind2count))

    return word2ind, log_probabilities


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: %s <test-filename> <vocab-filename> [smoothing-factor]" % sys.argv[0])
        sys.exit(1)

    test_filename = sys.argv[1]
    unigram_filename = sys.argv[2]
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    print('Using smoothing factor: ', alpha)

    word2ind, log_probs = read_unigram(unigram_filename, alpha)

    words_num = 0
    last_read_word = '<eon>'
    tot_log_prob = 0

    with open(test_filename, 'r') as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) == 0:
                tot_log_prob += log_probs[word2ind['<eon>']]
                words_num += 1
            else:
                for tok in toks:
                    tot_log_prob += log_probs[word2ind[tok]]
                tot_log_prob += log_probs[word2ind['<eos>']]
                words_num += len(toks)+1

    print('Read words:', words_num)
    print('perp:',np.exp(-tot_log_prob/words_num))

