import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_loader import *
from model.network import *

def cli_def():
    parser = argparse.ArgumentParser(description='Pytorch meta test example of network architectures for MLR-SNet')
    parser.add_argument('--network', type=str, default='shufflenetv2', choices=['shufflenetv2', 'mobilenetv2', 'nasnet'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--num-epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--times', type=int, default=0)
    return parser


def meta_test(network_name, dataset, num_epoch, batch_size, optim_name, lr, momentum, wd, model_loc, seed):

    if not os.path.isdir('result'):
        os.mkdir('result')
    save_path = './result/meta-test_' + str(args.network) + '_' + str(dataset)
    tr_loss = []
    t_loss = []
    tr_acc = []
    t_acc = []

    # We are using cuda for training - no point trying out on CPU for ResNet
    device = torch.device("cuda")

    net = build_network(network_name, dataset)
    net.to(device)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    # assign argparse parameters
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_accuracy = 0.0
    lr_data = []
    train_data, test_data = data_loader(dataset, batch_size)

    iter_len = len(train_data.dataset)
    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_loss, train_acc = compute_loss_accuracy(net, train_data, criterion, device)
    print('Initial training loss is %.3f' % train_loss)

    gamma = abs((train_loss ** 0.5 * np.log(train_loss * num_classes) / num_classes ** 0.25) / 4)
    print('First gamma is %.3f' % gamma)

    mlr_snet = MLRSNet(1, 50).to(device)
    print(mlr_snet)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    model_dict = torch.load('./mlr_snet 1.pth')
    mlr_snet.load_state_dict(model_dict)

    for epoch in range(num_epoch):
        train_correct = 0
        train_loss = 0

        if epoch == args.num_epoch // 3:
            net_path = './mlr_snet 100.pth'
            
            model_dict = torch.load(net_path)
            mlr_snet.load_state_dict(model_dict)
            mlr_snet = mlr_snet.to(device)
            
            train_loss, train_acc = compute_loss_accuracy(net, train_data, criterion, device)
            gamma = abs((train_loss ** 0.5 * np.log(train_loss * num_classes) / num_classes ** 0.25) / 4)

            print('Second gamma is %.3f' % gamma)

        if epoch==args.num_epoch // 3 * 2:
            net_path = './mlr_snet 200.pth'
            
            model_dict = torch.load(net_path)
            mlr_snet.load_state_dict(model_dict)
            mlr_snet = mlr_snet.to(device)
            
            train_loss, train_acc = compute_loss_accuracy(net, train_data, criterion, device)
            gamma = abs((train_loss ** 0.5 * np.log(train_loss * num_classes) / num_classes ** 0.25) / 4)
            
            print('Third gamma is %.3f' % gamma)

        for i, (inputs, labels) in enumerate(train_data):
            mlr_snet.reset_lstm(keep_states=(epoch + i) > 0, device=device)
            net.train()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * labels.size(0)

            train_pred = outputs.argmax(1)
            train_correct += train_pred.eq(labels).sum().item()

            loss_net = loss.unsqueeze(0)
            with torch.no_grad():
                lr_model = mlr_snet(loss_net)

            lr_model = float(lr_model.data) * gamma
            lr_data.append(lr_model)

            for group in optimizer.param_groups:
                group['lr'] = lr_model

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_acc = 100.0 * (train_correct / iter_len)
        val_loss, val_acc = compute_loss_accuracy(net, test_data, criterion, device)

        tr_loss.append(train_loss / iter_len)
        t_loss.append(val_loss)
        tr_acc.append(train_acc)
        t_acc.append(val_acc)
        torch.save({'train_acc': tr_acc, 'test_acc': t_acc,
                    'train_loss': tr_loss, 'test_loss': t_loss,
                    'lr': lr_data}, save_path)
        print('train loss is : %.4f' % (train_loss / iter_len))
        print('test loss is: %.4f' % val_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(net.state_dict(), model_loc)

        print('train_accuracy at epoch :{} is : {}'.format(epoch, train_acc))
        print('val_accuracy at epoch :{} is : {}'.format(epoch, val_acc))
        print('best val_accuracy is : {}'.format(best_val_accuracy))

        print('learning_rate after epoch :{} is : {}'.format(epoch, lr_data[-1]))


if __name__ == '__main__':
    args = cli_def().parse_args()
    print(args)

    meta_test(args.network, args.dataset, args.num_epoch, args.batch_size, args.optimizer, args.lr, args.momentum,
                    args.wd, args.model_loc, args.seed)
