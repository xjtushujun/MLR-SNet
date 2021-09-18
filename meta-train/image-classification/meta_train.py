import argparse
import os

import torch.optim as optim

from data_loader import build_dataset
from module import MLRSNet, MetaSGD
from meta_image_model import *
from utils import *


def cli_def():
    parser = argparse.ArgumentParser(description='Pytorch meta training example of image classification for MLR-SNet')
    parser.add_argument('--network', type=str, choices=['resnet', 'wideresnet'], default='resnet')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--num-epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0, help='momentum of sgd')
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of MLR-SNet\'s optimizer')
    parser.add_argument('--t_val', type=int, default=300, metavar='N',
                        help='updating mlr-snet every t_val batches (default: 300)')
    return parser


def meta_train():
    args = cli_def().parse_args()
    print(args)
    network = args.network
    dataset = args.dataset
    batch_size = args.batch_size
    lr = args.lr
    num_epoch = args.num_epoch

    if not os.path.isdir('result'):
        os.mkdir('result')
    save_path = './result/meta-train_' + network + '_' + dataset
    tr_loss = []
    t_loss = []
    tr_acc = []
    t_acc = []
    lr_save = []

    # We are using cuda for training - no point trying out on CPU for ResNet
    device = torch.device("cuda")

    if dataset == 'cifar10':
        num_classes = 10
    if dataset == 'cifar100':
        num_classes = 100

    model = build_network(network, num_classes)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.params()])))
    model.to(device).apply(init_weights)

    mlr_snet = MLRSNet(1, 50).to(device)
    print(mlr_snet)


    # assign argparse parameters
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_accuracy = 0.0

    num_meta = 1000
    train_data, meta_data,test_data = build_dataset(dataset, num_meta, batch_size)
    print(len(train_data),len(meta_data),len(test_data))

    train_loss, train_acc = compute_loss_accuracy(model, train_data, criterion, device)
    print('Initial training loss is %.3f' % train_loss)

    gamma = (train_loss**0.5*np.log(train_loss*num_classes)/num_classes**0.25)/4
    print('Gamma is %.3f' % gamma)

    optimizer_vnet = torch.optim.Adam(mlr_snet.params(), lr = lr, weight_decay=1e-4)
    optimizer = optim.SGD(model.params(), lr=1, momentum=args.momentum, weight_decay=args.wd)

    meta_data_iter = iter(meta_data)
    for epoch in range(num_epoch):
        train_correct = 0
        train_loss = 0

        for i, (inputs, labels) in enumerate(train_data):
            model.train()
            mlr_snet.reset_lstm(keep_states=(epoch+i)>0, device=device)
            inputs, labels = inputs.to(device), labels.to(device)
            if (i+1) % args.t_val == 0:

                meta_model = build_network(network, num_classes)
                meta_model.to(device)
                meta_model.load_state_dict(model.state_dict())
                meta_model.train()

                outputs = meta_model(inputs)
                loss = criterion(outputs, labels)
                loss = loss.unsqueeze(0)

                meta_model.zero_grad()
                grads = torch.autograd.grad(loss, (meta_model.params()), create_graph=True)
                input = loss
                lr_ = mlr_snet(input.unsqueeze(0))

                optimizer_metamodel = MetaSGD(meta_model)
                optimizer_metamodel.load_state_dict(optimizer.state_dict())
                optimizer_metamodel.step(lr=lr_ * gamma, grad=grads)

                del grads

                try:
                    inputs_val, targets_val = next(meta_data_iter)
                except StopIteration:
                    meta_data_iter = iter(meta_data)
                    inputs_val, targets_val = next(meta_data_iter)
                inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                y_g_hat = meta_model(inputs_val)
                l_g_meta = criterion(y_g_hat, targets_val.long())

                optimizer_vnet.zero_grad()
                l_g_meta.backward()
                optimizer_vnet.step()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            input = loss.unsqueeze(0)
            with torch.no_grad():
                new_lr = mlr_snet(input)

            new_lr = float(new_lr.data)*gamma
            lr_save.append(new_lr)

            for group in optimizer.param_groups:
                group['lr'] = new_lr

            train_loss += loss.item() * labels.size(0)

            train_pred = outputs.argmax(1)
            train_correct += train_pred.eq(labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        train_acc = 100.0 * (train_correct / len(train_data.dataset))
        val_loss, val_acc = compute_loss_accuracy(model, test_data, criterion, device)

        tr_loss.append(train_loss / len(train_data.dataset))
        t_loss.append(val_loss)
        tr_acc.append(train_acc)
        t_acc.append(val_acc)
        torch.save({'train_acc': tr_acc, 'test_acc': t_acc,
                    'train_loss': tr_loss, 'test_loss': t_loss, 'lr': lr_save},
                   save_path)
        print('train loss is : %.4f' % (train_loss / len(train_data.dataset)))
        print('test loss is: %.4f' % val_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

        torch.save(mlr_snet.state_dict(),'./result/mlr_snet %d.pth' % (epoch+1))

        print('train_accuracy at epoch :{} is : {}'.format(epoch, train_acc))
        print('val_accuracy at epoch :{} is : {}'.format(epoch, val_acc))
        print('best val_accuracy is : {}'.format(best_val_accuracy))

        cur_lr = 0.0
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate after epoch :{} is : {}'.format(epoch, cur_lr))


if __name__ == '__main__':

    meta_train()
