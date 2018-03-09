import torch, torchvision

from .utils import apply_mode

def alexnet(num_classes=1000, pretrained=4, mode=('freeze', 'fine-tune')):
    neural_network = torchvision.models.alexnet(pretrained=True)
    neural_network.classifier._modules['6'] = torch.nn.Linear(4096, num_classes)

    if pretrained == -1:
        return torchvision.models.alexnet(pretrained=False, num_classes=num_classes)

    layers = list(neural_network.features.children()) + list(neural_network.classifier.children())
    level = [-1, 7, 12, 15, 18][pretrained]

    for i, layer in enumerate(layers):
        if i <= level:
            apply_mode(layer, mode[0])
        else:
            apply_mode(layer, mode[1])

    return neural_network

# pretrained:
#  -1 - original model (no pretrained initialization)
#   0 - no transfered
#   1 - half feature transfered
#   2 - feature transfered
#   3 - feature + half classification transfered
#   4 - all transfered except final layer

# alexnet:
# layer number | pretrained level | layer name
#                       0           --- features
#      0                              Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#      1                              ReLU(inplace=True),
#      2                              MaxPool2d(kernel_size=3, stride=2),
#      3                              Conv2d(64, 192, kernel_size=5, padding=2),
#      4                              ReLU(inplace=True),
#      5                              MaxPool2d(kernel_size=3, stride=2),
#      6                              Conv2d(192, 384, kernel_size=3, padding=1),
#      7                1             ReLU(inplace=True),
#      8                              Conv2d(384, 256, kernel_size=3, padding=1),
#      9                              ReLU(inplace=True),
#     10                              Conv2d(256, 256, kernel_size=3, padding=1),
#     11                              ReLU(inplace=True),
#     12                2             MaxPool2d(kernel_size=3, stride=2),
#                                   --- classifier
#     13                              Dropout(),
#     14                              Linear(256 * 6 * 6, 4096),
#     15                3             ReLU(inplace=True),
#     16                              Dropout(),
#     17                              Linear(4096, 4096),
#     18                4             ReLU(inplace=True),
#     19                              Linear(4096, num_classes),
