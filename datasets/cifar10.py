import torch
from torchvision import transforms, datasets

from .sampler import BalancedSubsetRandomSampler, RandomSampler


def cifar10(download=False, num_workers=2, batch_size=64, img_size=(224,224), sample_per_class=(-1, -1)):
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])

    # train data
    train_data = datasets.CIFAR10(root='datasets/CIFAR10', train=True,
                                            download=download, transform=transform)
    if sample_per_class[0] == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = BalancedSubsetRandomSampler(train_data, sample_per_class[0], 10)

    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    # test data
    test_data = datasets.CIFAR10(root='datasets/CIFAR10', train=False,
                                            download=download, transform=transform)
    if sample_per_class[1] == -1:
        test_sampler = RandomSampler(test_data)
    else:
        test_sampler = BalancedSubsetRandomSampler(test_data, sample_per_class[1], 10)

    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)


    return train_loader, test_loader
