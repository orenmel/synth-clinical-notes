'''
Generates synthetic notes based on a smoothed unigram distribution
'''

import sys
import numpy as np

def read_unigram(path, alpha):
    ind2word = []
    ind2count = []
    with open(path) as f:
        for line in f:
            word, count = line.strip().split()
            count = float(count) + alpha # lidstone smoothing
            ind2word.append(word)
            ind2count.append(count)

    probabilities = np.asarray(ind2count)/sum(ind2count)

    return ind2word, probabilities


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: %s <output-filename> <vocab-filename> <size> [smoothing-factor]" % sys.argv[0])
        sys.exit(1)

    output_filename = sys.argv[1]
    unigram_filename = sys.argv[2]
    size = int(sys.argv[3])
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    print('Using smoothing factor: ', alpha)

    ind2word, unigram_probs = read_unigram(unigram_filename, alpha)

    words_num = 0
    last_sampled_word = '<eon>'

    with open(output_filename, 'w') as f:

        while words_num < size:

            sampled_word_ind = np.random.choice(unigram_probs.size, size=None, replace=True, p=unigram_probs)
            sampled_word = ind2word[sampled_word_ind]
            if sampled_word == '<eon>':
                if last_sampled_word == '<eos>':
                    f.write('\n')
                elif last_sampled_word == '<eon>':
                    pass
                else:
                    f.write('\n\n')
            elif sampled_word == '<eos>':
                if last_sampled_word != '<eos>' and last_sampled_word != '<eon>':
                    f.write('\n')
            else:
                if last_sampled_word != '<eos>' and last_sampled_word != '<eon>':
                    f.write(' ')
                f.write(sampled_word)
                words_num += 1

            last_sampled_word = sampled_word
            if words_num % 1000000 == 0:
                print('Generated %d words so far.' % words_num)

