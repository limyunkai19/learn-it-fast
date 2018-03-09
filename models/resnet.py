import torch, torchvision

from .utils import apply_mode

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def resnet(resnet_name='resnet50', num_classes=1000, pretrained=4, mode=('freeze', 'fine-tune')):
    resnet_model = torchvision.models.__dict__[resnet_name]
    if pretrained == -1:
        return resnet_model(pretrained=False, num_classes=num_classes)

    neural_network = resnet_model(pretrained=True)
    in_features = neural_network.fc.in_features
    neural_network.fc = torch.nn.Linear(in_features, num_classes)

    layers = [layer for layer in neural_network.children()]
    level = [-1, 4, 5, 6, 8][pretrained]

    for i, layer in enumerate(layers):
        if i <= level:
            apply_mode(layer, mode[0])
        else:
            apply_mode(layer, mode[1])

    return neural_network

def resnet18(**kwargs): return resnet(resnet_name='resnet18', **kwargs)
def resnet34(**kwargs): return resnet(resnet_name='resnet34', **kwargs)
def resnet50(**kwargs): return resnet(resnet_name='resnet50', **kwargs)
def resnet101(**kwargs): return resnet(resnet_name='resnet101', **kwargs)
def resnet152(**kwargs): return resnet(resnet_name='resnet152', **kwargs)

# pretrained:
#  -1 - original model (no pretrained weight initialization)
#   0 - not transfered
#   1 - transfer until layer 1 (inclusive)
#   2 - transfer until layer 2 (inclusive)
#   3 - transfer until layer 3 (inclusive)
#   4 - all transfered except final layer

# resnet:
# layer number | pretrained level | layer name
#                       0
#      0                              conv1
#      1                              bn1
#      2                              relu
#      3                              maxpool
#      4                1             layer1
#      5                2             layer2
#      6                3             layer3
#      7                              layer4
#      8                4             avgpool
#      9                              fc
