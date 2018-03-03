import torch
from torchvision import transforms

from .torchvision_mnist_master import MNIST

def mnist(download=True, num_workers=2, batch_size=64, img_size=(224,224)):
    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

    train_data = MNIST(root='datasets/MNIST', train=True,
                                            download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    test_data = MNIST(root='datasets/MNIST', train=False,
                                            download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    return train_loader, test_loader