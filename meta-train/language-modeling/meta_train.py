# Some part of the code was referenced from below.
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import time
from data_utils import Dictionary, Corpus
from model.meta_language_model import RNNLM, Transformer
from ../../meta_module.module import MLRSNet, MetaSGD
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch meta training example of language modeling for MLR-SNet ')
parser.add_argument('--model', default='lstm', type=str,
                    help='language model (lstm [default])')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--seq_len', type=int, default=35, metavar='N',
                    help='sequence length (default: 35)')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate of language model')
parser.add_argument('--momentum', default=0.0, type=float, help='momentum of sgd')
parser.add_argument('--wd', type=float, default=5e-6)
parser.add_argument('--max_epoch', type=int, default=150, metavar='N',
                    help='number of training epoch (default: 150)')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--embed_size', type=int, default=512, metavar='N',
                    help='embedding size of word (default: 512)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size of LSTM (default: 512)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--t_val', type=int, default=300, metavar='N',
                    help='updating mlr-snet every t_val batches (default: 300)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
args = parser.parse_args()
print(args)


def build_dataset(train_bs, test_bs):
    path = '../../data/ptb'
    train_path = os.path.join(path, 'train.txt')
    val_path = os.path.join(path, 'valid.txt')
    test_path = os.path.join(path, 'test.txt')

    corpus = Corpus([train_path, val_path, test_path], train_bs, train_bs, test_bs)
    print('Data is loaded.')

    return corpus


def build_model(num_layers, hidden_size, vocabulary_size, embedding_size, device):
    # use a multi-layer LSTM
    model = RNNLM(vocabulary_size, embedding_size, hidden_size=hidden_size, num_layers=num_layers)

    model.to(device)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.params()])))

    return model


def detach(states):
    return [state.detach() for state in states]


def clip_gradient(grads, max_norm, type=2.0):
    total_norm = 0
    for grad in grads:
        norm = grad.data.norm(type)
        total_norm += norm.item() ** type
    total_norm = total_norm ** (1. / type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.data.mul_(clip_coef)


def test(model, data, batch_size, sequence_length, vocab_size, criterion, device):
    model.eval()
    test_num_batches = data.size(0) // sequence_length
    states_test = (torch.zeros(args.num_layers, batch_size, args.hidden_size).to(device),
                   torch.zeros(args.num_layers, batch_size, args.hidden_size).to(device))
    test_loss = 0.0
    for k in range(0, data.size(0) - sequence_length, sequence_length):
        inputs = data[k:k + sequence_length].to(device)
        targets = data[(k + 1):(k + 1) + sequence_length].to(device)

        with torch.no_grad():
            states_test = detach(states_test)
            outputs, states_test = model(inputs, states_test)
            loss = criterion(outputs, targets.reshape(-1))
            test_loss += loss

    perp = np.exp(test_loss.item() / test_num_batches)
    print('Test Perplexity: {:5.2f}'.format(perp))
    print('-' * 60)
    return perp


def main():

    # device and hyperparameter setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_bs = args.batch_size
    test_bs = 1
    sequence_length = args.seq_len
    embedding_size = args.embed_size
    max_epoch = args.max_epoch

    # savings
    lr = []
    train_perplexity = []
    test_perplexity = []
    if not os.path.isdir('result'):
        os.mkdir('result')
    save_path = './result/meta-train_ptb_' + str(args.num_layers) + 'layer-lstm'

    # build dataset
    data = build_dataset(train_bs, test_bs)
    vocab_size = len(data.dictionary)
    print('Vocabulary size: %d' % vocab_size)
    num_batches = data.train.size(0) // sequence_length
    meta_num_batches = data.val.size(0) // sequence_length

    dumb_loss = np.log(vocab_size)

    # build model
    n_layers = args.num_layers
    h_size = args.hidden_size
    model = build_model(n_layers, h_size, vocab_size, embedding_size, device)

    # build optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = None
    optimizer = optim.SGD(model.params(), lr=1, momentum=args.momentum, weight_decay=args.wd)
    mlr_snet = MLRSNet(1,50)
    print(mlr_snet)
    lr_optimizer = optim.Adam(mlr_snet.params(), lr=args.lr, weight_decay=1e-4)
    init_train_perp = test(model, data.train, train_bs, sequence_length, vocab_size, criterion, device)
    print('Initial training perplexity is %.3f' % init_train_perp)

    gamma = (test_perp**0.5*np.log(test_perp*vocab_size)/vocab_size**0.25)/4
    print('Gamma is %.3f' % gamma)

    # training
    num = 0  # meta data batch index
    best_prep=10000
    for epoch in range(max_epoch):
        model.train()
        # Set initial hidden and cell states
        states = (torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device),
                  torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device))

        for i in range(0, data.train.size(0) - sequence_length, sequence_length):
            mlr_snet.reset_lstm(keep_states=(epoch+i) > 0, device=device)
            step = (i + 1) // sequence_length
            # Get mini-batch inputs and targets
            inputs = data.train[i:i + sequence_length].to(device)
            targets = data.train[(i + 1):(i + 1) + sequence_length].to(device)

            # update mlr_snet
            if step % args.t_val == 0:

                meta_model = build_model(n_layers, h_size, vocab_size, embedding_size, device)
                meta_model.load_state_dict(model.state_dict())
                meta_model.train()

                state = detach(states)
                outputs, _ = meta_model(inputs, state)
                loss_ = criterion(outputs, targets.reshape(-1))

                meta_model.zero_grad()
                grads = torch.autograd.grad(loss_, (meta_model.params()), create_graph=True)
                input = loss_ / dumb_loss
                lr_ = mlr_snet(input.unsqueeze(0))

                optimizer_metamodel = MetaSGD(meta_model)
                optimizer_metamodel.load_state_dict(optimizer.state_dict())
                clip_gradient(grads, args.clip)
                optimizer_metamodel.step(lr=lr_*gamma, grad=grads)

                del grads

                if num == 0 or num//sequence_length > meta_num_batches - 1:
                    meta_states = (torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device),
                                   torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device))
                    num = 0

                meta_inputs = data.val[num:num + sequence_length].to(device)
                meta_targets = data.val[(num + 1):(num + 1) + sequence_length].to(device)
                num += sequence_length

                # Forward pass
                meta_states = detach(meta_states)
                meta_outputs, meta_states = meta_model(meta_inputs, meta_states)
                meta_loss = criterion(meta_outputs, meta_targets.reshape(-1))

                # Backward and optimize
                lr_optimizer.zero_grad()
                meta_loss.backward()
                lr_optimizer.step()

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.params(), args.clip)

            input = loss / dumb_loss
            new_lr = mlr_snet(input.unsqueeze(0))
            for group in optimizer.param_groups:
                group['lr'] = float(new_lr * gamma)
            optimizer.step()

            lr.append(optimizer.param_groups[0]['lr'])

            if step % 100 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}, input: {:.4f}, lr: {:.4f}'
                      .format(epoch + 1, max_epoch, step, num_batches, loss.item(), np.exp(loss.item()),
                              input.item(), gamma*new_lr.item()))

        train_perplexity.append(np.exp(loss.item()))

        # testing
        test_perp = test(model, data.test, test_bs, sequence_length, vocab_size, criterion, device)
        test_perplexity.append(test_perp)

        if best_prep > test_perp:
            best_prep = test_perp

        print('='*85)
        print('Best Test Perplexity: {:5.2f}'.format(best_prep))

        # saving
        torch.save({'train_perplexity': train_perplexity, 'test_perplexity': test_perplexity, 'lr': lr},
                   save_path)
        torch.save(mlr_snet.state_dict(), './result/mlr-snet %d.pth' % (epoch + 1))


if __name__ == '__main__':
    main()
