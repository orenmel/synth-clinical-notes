import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)







class Corpus(object):
    def __init__(self, root_path = None, text_path = None, dict_path = None, read_test=False):

        if dict_path == None:
            self.dictionary = Dictionary()
        else:
            self.load_dict(dict_path)


        if text_path != None:
            # self.tokenize(path+'.all.txt') # this is just to build a dictionary consistent with training
            self.text = self.tokenize(text_path)
        else:
            # this is just to build a dictionary that includes the words in all.txt
            # all = os.path.join(path, 'all.txt')
            # if os.path.exists(all):
            #     print('loading all.txt...')
            #     self.tokenize(all)

            self.train = self.tokenize(os.path.join(root_path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(root_path, 'valid.txt'))

            if read_test:
                print('YES LOADING TEST TO DICT') # use run-time param to control this
                self.test = self.tokenize(os.path.join(root_path, 'test.txt'))
        print("Vocab size:", len(self.dictionary.word2idx))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def dump_dict(self, path):
        with open(path, 'w') as f:
            for word in self.dictionary.idx2word:
                f.write(word + '\n')

    def load_dict(self, path):
        self.dictionary = Dictionary()
        with open(path, 'r') as f:
            for line in f:
                word = line.strip()
                self.dictionary.add_word(word)


if __name__ == '__main__':

    import sys

    if len(sys.argv) < 2:
        print("Usage: %s <input-text-filename>" % sys.argv[0])
        sys.exit(1)

    text_filename = sys.argv[1]
    dict_filename = text_filename+'.vocab'

    corpus = Corpus(root_path=None, text_path=text_filename)
    corpus.dump_dict(dict_filename)