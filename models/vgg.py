import torch, torchvision

from .utils import apply_mode

__all__ = [ 'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']


def vgg(vgg_name='vgg16_bn', num_classes=1000, pretrained=4, mode=('freeze', 'fine-tune')):
    vgg_model = torchvision.models.__dict__[vgg_name]
    if pretrained == -1:
        return vgg_model(pretrained=False, num_classes=num_classes)

    neural_network = vgg_model(pretrained=True)
    neural_network.classifier._modules['6'] = torch.nn.Linear(4096, num_classes)

    layers_features = neural_network.features.children()
    layers_classifier = neural_network.classifier.children()
    level = [-1, 2, 4, 7, 10][pretrained]

    num_max_pool = 0
    for layer in layers_features:
        if num_max_pool <= level:
            apply_mode(layer, mode[0])
        else:
            apply_mode(layer, mode[1])

        if isinstance(layer, torch.nn.MaxPool2d):
            num_max_pool += 1

    for i, layer in enumerate(layers_classifier, 5):
        if i <= level:
            apply_mode(layer, mode[0])
        else:
            apply_mode(layer, mode[1])

    return neural_network

def vgg11(**kwargs): return vgg(vgg_name='vgg11', **kwargs)
def vgg13(**kwargs): return vgg(vgg_name='vgg13', **kwargs)
def vgg16(**kwargs): return vgg(vgg_name='vgg16', **kwargs)
def vgg19(**kwargs): return vgg(vgg_name='vgg19', **kwargs)
def vgg11_bn(**kwargs): return vgg(vgg_name='vgg11_bn', **kwargs)
def vgg13_bn(**kwargs): return vgg(vgg_name='vgg13_bn', **kwargs)
def vgg16_bn(**kwargs): return vgg(vgg_name='vgg16_bn', **kwargs)
def vgg19_bn(**kwargs): return vgg(vgg_name='vgg19_bn', **kwargs)

# pretrained:
#  -1 - original model (no pretrained weight initialization)
#   0 - not transfered
#   1 - half feature transfered
#   2 - feature transfered
#   3 - feature + half classification transfered
#   4 - all transfered except final layer

# observation:
#   all vgg has features and classifier layers
#   all vgg has 5 max pooling layer in features layers
#   all vgg has same classifier layers

# vgg:
# layer number | pretrained level | layer name
#                       0           --- features
#                                     Conv + (batch norm) + relu
#                                     .
#        0                            MaxPool2d
#                                     Conv + (batch norm) + relu
#                                     .
#        1                            MaxPool2d
#                                     Conv + (batch norm) + relu
#                                    .
#        2              1             MaxPool2d
#                                     Conv + (batch norm) + relu
#                                     .
#        3                            MaxPool2d
#                                     Conv + (batch norm) + relu
#                                     .
#        4              2             MaxPool2d
#                                   --- classifier
#        5                            Linear
#        6                            ReLU
#        7              3             Dropout
#        8                            Linear
#        9                            ReLU
#       10              4             Dropout
#       11                            Linear
