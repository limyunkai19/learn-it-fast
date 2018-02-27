import torch, torchvision

def alexnet(num_classes):
    neural_network = torchvision.models.alexnet(pretrained=True)
    neural_network.classifier._modules['6'] = torch.nn.Linear(4096, num_classes)
    return neural_network
