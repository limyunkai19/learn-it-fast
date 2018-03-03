import torch
from torchvision import transforms, datasets

def cifar10(download=True, num_workers=2, batch_size=64, img_size=(224,224)):
    transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])

    train_data = datasets.CIFAR10(root='datasets/CIFAR10', train=True,
                                            download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    test_data = datasets.CIFAR10(root='datasets/CIFAR10', train=False,
                                            download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    return train_loader, test_loader