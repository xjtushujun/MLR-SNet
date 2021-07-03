import torch
from torchvision import datasets, transforms

data_loc = '../data/'


def data_loader(dataset_name, batch_size):
    if dataset_name == 'cifar10':
        return cifar_10_data_loader(batch_size)
    elif dataset_name == 'cifar100':
        return cifar_100_data_loader(batch_size)
    elif dataset_name == 'svhn':
        return svhn_data_loader(batch_size)
    else:
        raise Exception('dataset is not supported')


def cifar_10_data_loader(batch_size):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def cifar_100_data_loader(batch_size):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    dataset = datasets.CIFAR100(root=data_loc, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    test_dataset = datasets.CIFAR100(root=data_loc, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def svhn_data_loader(batch_size):

    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root=data_loc, split='train', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        ),
        batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root=data_loc, split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        ),
        batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), 100.0 * (correct / len(data_loader.dataset))
