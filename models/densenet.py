import torch, torchvision

from .utils import apply_mode

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']


def densenet(densenet_name='densenet121', num_classes=1000, pretrained=4, mode=('freeze', 'fine-tune')):
    densenet_model = torchvision.models.__dict__[densenet_name]
    if pretrained == -1:
        neural_network = densenet_model(pretrained=False, num_classes=num_classes)
        neural_network.meta = {
            'base_model': densenet_name,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'mode': mode
        }
        return neural_network

    neural_network = densenet_model(pretrained=True)
    in_features = neural_network.classifier.in_features
    neural_network.classifier = torch.nn.Linear(in_features, num_classes)

    layers = [layer for layer in neural_network.features.children()]
    level = [-1, 5, 7, 9, 11][pretrained]

    for i, layer in enumerate(layers):
        if i <= level:
            apply_mode(layer, mode[0])
        else:
            apply_mode(layer, mode[1])

    neural_network.meta = {
        'base_model': densenet_name,
        'num_classes': num_classes,
        'pretrained': pretrained,
        'mode': mode
    }
    return neural_network

def densenet121(**kwargs): return densenet(densenet_name='densenet121', **kwargs)
def densenet169(**kwargs): return densenet(densenet_name='densenet169', **kwargs)
def densenet201(**kwargs): return densenet(densenet_name='densenet201', **kwargs)
def densenet161(**kwargs): return densenet(densenet_name='densenet161', **kwargs)

# pretrained:
#  -1 - original model (no pretrained weight initialization)
#   0 - not transfered
#   1 - transfer until denseblock and transition 1 (inclusive)
#   2 - transfer until denseblock and transition 2 (inclusive)
#   3 - transfer until denseblock and transition 3 (inclusive)
#   4 - all transfered except final layer

# densenet:
# layer number | pretrained level | layer name
#                       0             features
#      0                                conv0
#      1                                norm0 - batch normalization
#      2                                relu0
#      3                                pool0
#      4                                denseblock1
#      5                1               transition1
#      6                                denseblock2
#      7                2               transition2
#      8                                denseblock3
#      9                3               transition3
#      10                               denseblock4
#      11               4               norm5 - batch normalization
#                                     classifier
#      12                               linear - fc
