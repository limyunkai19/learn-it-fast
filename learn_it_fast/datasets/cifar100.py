import torch
from torchvision import transforms, datasets

from .sampler import BalancedSubsetRandomSampler, RandomSampler


def cifar100(download=False, num_workers=2, batch_size=64,
                        img_size=224, sample_per_class=(-1, -1), data_augmentation=False):
    base_transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                ])

    # train data
    if data_augmentation:
        train_transform = transforms.Compose([
                              transforms.RandomResizedCrop(224, scale=(0.25, 1)),
                              transforms.RandomHorizontalFlip(),
                              base_transform
                          ])
    else:
        train_transform = base_transform

    train_data = datasets.CIFAR100(root='datasets/CIFAR100', train=True,
                                            download=download, transform=train_transform)
    if sample_per_class[0] == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = BalancedSubsetRandomSampler(train_data, sample_per_class[0], 100)

    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    # test data
    test_data = datasets.CIFAR100(root='datasets/CIFAR100', train=False,
                                            download=download, transform=base_transform)
    if sample_per_class[1] == -1:
        test_sampler = RandomSampler(test_data)
    else:
        test_sampler = BalancedSubsetRandomSampler(test_data, sample_per_class[1], 100)

    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)


    return train_loader, test_loader
