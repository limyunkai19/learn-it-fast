import torch, torchvision

from .utils import apply_mode

def inception_v3(num_classes=1000, pretrained=4, mode=('freeze', 'fine-tune')):
    if pretrained == -1:
        return torchvision.models.inception_v3(pretrained=False, num_classes=num_classes)

    neural_network = torchvision.models.inception_v3(pretrained=True)
    neural_network.fc = torch.nn.Linear(2048, num_classes)

    layers = [layer for layer in neural_network.children()]
    level = [-1, 4, 7, 13, 16][pretrained]

    for i, layer in enumerate(layers):
        if i <= level:
            apply_mode(layer, mode[0])
        else:
            apply_mode(layer, mode[1])

    return neural_network

# pretrained:
#  -1 - original model (no pretrained weight initialization)
#   0 - not transfered
#   1 - approx 25% transfered
#   2 - approx 50% transfered
#   3 - approx 75% transfered
#   4 - all transfered except final layer

# inception_v3:
# layer number | pretrained level | layer name
#                       0
#      0                              Conv2d_1a_3x3
#      1                              Conv2d_2a_3x3
#      2                              Conv2d_2b_3x3
#      3                              Conv2d_3b_1x1
#      4                1             Conv2d_4a_3x3
#      5                              Mixed_5b
#      6                              Mixed_5c
#      7                2             Mixed_5d
#      8                              Mixed_6a
#      9                              Mixed_6b
#     10                              Mixed_6c
#     11                              Mixed_6d
#     12                              Mixed_6e
#     13                3             AuxLogits
#     14                              Mixed_7a
#     15                              Mixed_7b
#     16                4             Mixed_7c
#     17                              fc
