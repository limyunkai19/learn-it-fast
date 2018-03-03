import torch
from torchvision import transforms

from .torchvision_mnist_master import EMNIST

def emnist(download=True, num_workers=2, batch_size=64, img_size=(224,224)):
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1751,), (0.3332,))
                ])

    train_data = EMNIST(root='datasets/EMNIST', split="balanced", train=True,
                                            download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    test_data = EMNIST(root='datasets/EMNIST', split="balanced", train=False,
                                            download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    return train_loader, test_loader