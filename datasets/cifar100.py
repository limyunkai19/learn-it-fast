import torch
from torchvision import transforms, datasets

def cifar100(download=True, num_workers=2, batch_size=64, img_size=(224,224)):
    transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                ])

    train_data = datasets.CIFAR100(root='./CIFAR100', train=True,
                                            download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    test_data = datasets.CIFAR100(root='./CIFAR100', train=False,
                                            download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)


    return train_loader, test_loader