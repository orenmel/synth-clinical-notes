# coding: utf-8
################################################################################
#
# This modules loads two language models and computes a measure of the
# difference in their likelihood prediction on a given input text.
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import math


import data



# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model1, model2, corpus, criterion, eval_batch_size, bptt):

    ntokens = len(corpus.dictionary)
    data_source = batchify(corpus.text, eval_batch_size)

    # Turn on evaluation mode which disables dropout.
    model1.eval()
    model2.eval()
    hidden1 = model1.init_hidden(eval_batch_size)
    hidden2 = model2.init_hidden(eval_batch_size)

    total_loss1 = 0
    total_loss2 = 0
    total_loss_diff = 0
    all_max_diff = 0
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt, evaluation=True)
        output1, hidden1 = model1(data, hidden1)
        output1_flat = output1.view(-1, ntokens)
        output2, hidden2 = model2(data, hidden2)
        output2_flat = output2.view(-1, ntokens)

        losses1 = criterion(output1_flat, targets).data
        losses2 = criterion(output2_flat, targets).data

        # losses_diff = torch.pow(losses1-losses2, 2)

        losses_diff = torch.abs(losses1 - losses2)

        print('%s\t%.2f\t%.2f\t%.2f' % (corpus.dictionary.idx2word[int(targets[0])].ljust(20), float(losses1[0]), losses2[0],losses_diff[0]) )

        max_diff = torch.max(losses_diff)
        if max_diff > all_max_diff:
            all_max_diff = max_diff

        total_loss1 += torch.sum(losses1)
        total_loss2 += torch.sum(losses2)
        total_loss_diff += torch.sum(losses_diff)

        hidden1 = repackage_hidden(hidden1)
        hidden2 = repackage_hidden(hidden2)

    total_eval_words = len(data_source)*eval_batch_size
    return total_loss_diff / total_eval_words, all_max_diff, total_loss1 / total_eval_words, total_loss2 / total_eval_words



parser = argparse.ArgumentParser(description='PyTorch Language Model Diff')

# Model parameters.
parser.add_argument('--corpus_vocab', type=str,
                    help='location of the corpus vocab file')
parser.add_argument('--corpus_eval', type=str,
                    help='location of the text corpus used to evaluate diff')
parser.add_argument('--model1', type=str, default='./model1.pt',
                    help='model 1 file')
parser.add_argument('--model2', type=str, default='./model2.pt',
                    help='model 2 file')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=1,
                    help='sequence length')
args = parser.parse_args()


# we use these settings for printing word-level diffs
assert(args.bptt == 1 and args.batch_size == 1)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.model1, 'rb') as f:
    model1 = torch.load(f)
with open(args.model2, 'rb') as f:
    model2 = torch.load(f)

criterion = nn.CrossEntropyLoss(reduce=False)

if args.cuda:
    model1.cuda()
    model2.cuda()
else:
    model1.cpu()
    model2.cpu()

corpus = data.Corpus(dict_path=args.corpus_vocab, text_path=args.corpus_eval)

print('Corpus vocab: ', len(corpus.dictionary))

mean_diff, max_diff, loss1, loss2 = evaluate(model1, model2, corpus, criterion, args.batch_size, args.bptt)


print('loss1 {:5.2f} | ppl1 {:8.2f}'.format(loss1, math.exp(loss1)))
print('loss2 {:5.2f} | ppl2 {:8.2f}'.format(loss2, math.exp(loss2)))
print('mean diff metric {:5.2f} max diff metric {:5.2f} '.format(mean_diff, max_diff))

