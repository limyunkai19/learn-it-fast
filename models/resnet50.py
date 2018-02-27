import torch, torchvision

def resnet50(num_classes):
    neural_network = torchvision.models.resnet50(pretrained=True)
    new_fc = nn.Linear(neural_network.fc.in_features, num_classes)
    neural_network.classifier._modules['6'] = new_fc
    return neural_network
