from .cifar10 import cifar10
from .cifar100 import cifar100
from .mnist import mnist
from .emnist import emnist

available_datasets = ['cifar10', 'cifar100', 'mnist', 'emnist']
num_classes = {'cifar10': 10, 'cifar100': 100, 'mnist': 10, 'emnist': 47}
