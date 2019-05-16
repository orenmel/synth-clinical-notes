'''
Loads dataset, counts vocabulary, dumps vocab to file
EO is used to count occurrences of end-of-sentence (end of line) end-of-note (an empty line).
in-vocab-filename is used to initialize the vocabulary with zero counts.
'''


import collections
import sys


def save_vocab(tok2count, filename):
    with open(filename, 'w') as f:
        for tok, count in tok2count.most_common():
            f.write("%s\t%d\n" % (tok, count))

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <dataset-filename> <out-vocab-filename> [EO] [in-vocab-filename]" % sys.argv[0])
        sys.exit(1)

    dataset_filename = sys.argv[1]
    vocab_filename = sys.argv[2]

    eo = (len(sys.argv) > 3) and (sys.argv[3] == 'EO')

    tok2count = collections.Counter()

    if len(sys.argv) > 4:
        in_vocab_filename = sys.argv[4]
        with open(in_vocab_filename, 'r') as f:
            for line in f:
                word = line.strip().split()[0]
                tok2count[word] = 0

    last_line_not_empty = False

    with open(dataset_filename, 'r') as fin:
        for line in fin:
            toks = line.strip().split()
            for t in toks:
                assert(not (eo and (t=='<eon>' or t=='<eos>')))
                tok2count[t] += 1
            if eo:
                if len(toks) > 0:
                    tok2count['<eos>'] += 1
                if len(toks) == 0 and last_line_not_empty:
                    tok2count['<eon>'] += 1
            last_line_not_empty = len(toks) > 0

    if eo:
        tok2count['<eon>'] += 1

    save_vocab(tok2count, vocab_filename)

