'''
Replaces rare words with <unk>
'''

import collections
import sys

MIN_FREQ = 3

def load_vocab(filename):
    tok2count = {}
    with open(filename, 'r') as f:
        for line in f:
            segs = line.strip().split()
            tok = segs[0]
            count = int(segs[1])
            tok2count[tok]=count
    return tok2count

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <dataset-filename> <vocab-filename> <output-filename> [min-freq]" % sys.argv[0])
        sys.exit(1)

    dataset_filename = sys.argv[1]
    vocab_filename = sys.argv[2]
    output_filename = sys.argv[3]

    if len(sys.argv) > 4:
        min_freq = int(sys.argv[4])
    else:
        min_freq = MIN_FREQ

    print('min_freq', min_freq)

    tok2count = load_vocab(vocab_filename)

    with open(dataset_filename, 'r') as fin, open(output_filename, 'w') as fout:
        for line in fin:
            toks = [tok if tok2count[tok] >= min_freq else '<unk>' for tok in line.strip().split()]
            fout.write(' '.join(toks) + '\n')
