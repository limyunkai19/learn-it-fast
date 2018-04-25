from .cifar10 import cifar10
from .cifar100 import cifar100
from .mnist import mnist
from .emnist import emnist
from .stl10 import stl10
from .folder import folder
from .folder2 import folder2

available_datasets = ['cifar10', 'cifar100', 'mnist', 'emnist', 'stl10']
num_classes = {'cifar10': 10, 'cifar100': 100, 'mnist': 10, 'emnist': 47, 'stl10': 10}
