# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging
import sys


import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default=None,
                    help='location of the data corpus')
# parser.add_argument('--train', type=str, default=None,
#                     help='location of the train data corpus (no eval)')
parser.add_argument('--vocab', type=str, default=None,
                    help='location of the corpus vocab file')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--lr_backoff', type=float, default=4.0,
                    help='lr reduce factor')
parser.add_argument('--lr_min', type=float, default=0.1,
                    help='lr is never reduced below this')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--epoch_size', type=int, default=0,
                    help='max number of words processed per epoch (0 means the entire train set)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default=None,
                    help='path to load a trained model (no training, just testing)')
parser.add_argument('--test', action='store_true',
                    help='run on test')
args = parser.parse_args()


log_file_name = args.load+'.apply.log' if args.load is not None else args.save+'.log'

logging.basicConfig(filename=log_file_name,level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


logging.info(' '.join(sys.argv))

if args.load is not None:
    logging.info("NOTICE: Loading and testing model. No training.")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################


# assert((args.data != None and args.train == None) or (args.data == None and args.train != None))
# corpus = data.Corpus(root_path=args.data, text_path=args.train, dict_path=args.vocab)
corpus = data.Corpus(root_path=args.data, dict_path=args.vocab, read_test=args.test)

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

#do_eval = args.data != None

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)

if args.load is not None:
    test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.load is not None:
    model = torch.load(args.load)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
        # return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(begin=0, end=0):

    if end==0:
        end = train_data.size(0) - 1
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)

    # print(end)
    # print(int(end))
    # print(train_data.size(0))

    for batch, i in enumerate(range(begin, end, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            # logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #         'loss {:5.2f} | ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // args.bptt, lr,
            #     elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, (end-begin) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if args.load is not None:
    val_loss = evaluate(val_data)
    logging.info('-' * 89)
    logging.info(' valid loss {:5.2f} | valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))

    if args.test:
        # Run on test data.
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)



else:
# Loop over epochs.
    lr = args.lr
    best_val_loss = None


    batches_per_epoch = int(args.epoch_size / args.batch_size)
    begin_batch = 0


    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            if args.epoch_size > 0:
                if begin_batch + batches_per_epoch > train_data.size(0)-1:
                    logging.info('Starting to read train data from the beginning after %d words.' % (begin_batch * args.batch_size))
                    begin_batch = 0
                train(begin=begin_batch, end=begin_batch + batches_per_epoch)
                begin_batch += batches_per_epoch
            else:
                train()

            val_loss = evaluate(val_data)
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging.info('-' * 89)
            if best_val_loss != None and math.exp(best_val_loss) - math.exp(val_loss) <  0.1:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                # lr /= 4.0
                if lr > args.lr_min:
                    lr = max(lr/args.lr_backoff, args.lr_min)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            # else:

    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')

    # Load the best saved model.
    #with open(args.save, 'rb') as f:
    #    model = torch.load(f)


