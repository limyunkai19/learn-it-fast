import torch
from torchvision import transforms, datasets
from torchvision.datasets.folder import find_classes

from .sampler import BalancedSubsetRandomSampler, RandomSampler


def folder(root, num_workers=2, batch_size=64, img_size=224, sample_per_class=-1, data_augmentation=False):
    base_transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    if data_augmentation:
        data_transform = transforms.Compose([
                              transforms.RandomResizedCrop(img_size, scale=(0.25, 1)),
                              transforms.RandomHorizontalFlip(),
                              base_transform
                          ])
    else:
        data_transform = base_transform

    data = datasets.ImageFolder(root=root, transform=data_transform)
    if sample_per_class == -1:
        data_sampler = RandomSampler(data)
    else:
        data_sampler = BalancedSubsetRandomSampler(data, sample_per_class, len(find_classes(root)[0]))

    data_loader = torch.utils.data.DataLoader(data,
                        batch_size=batch_size, sampler=data_sampler, num_workers=num_workers)


    return data_loader
