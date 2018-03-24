import torch
from torchvision import transforms

from .torchvision_mnist_master import EMNIST
from .sampler import BalancedSubsetRandomSampler, RandomSampler


def emnist(download=False, num_workers=2, batch_size=64, img_size=(224,224), sample_per_class=(-1, -1)):
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1751, 0.1751, 0.1751), (0.3332, 0.3332, 0.3332))
                ])

    # train data
    train_data = EMNIST(root='datasets/EMNIST', split="balanced", train=True,
                                            download=download, transform=transform)
    if sample_per_class[0] == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = BalancedSubsetRandomSampler(train_data, sample_per_class[0], 47)

    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    # test data
    test_data = EMNIST(root='datasets/EMNIST', split="balanced", train=False,
                                            download=download, transform=transform)
    if sample_per_class[1] == -1:
        test_sampler = RandomSampler(test_data)
    else:
        test_sampler = BalancedSubsetRandomSampler(test_data, sample_per_class[1], 47)

    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)


    return train_loader, test_loader
