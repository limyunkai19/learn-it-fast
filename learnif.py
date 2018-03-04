import argparse

import torch, torchvision
import torch.nn as nn
import torch.optim as optim

import models, datasets
from utils import model_fit

# Command line argument
parser = argparse.ArgumentParser(description='Transfer Learning Train')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training and testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='use pretrained model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed replicable result (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.pretrained:
    cnn = models.alexnet(10)
else:
    cnn = torchvision.models.alexnet(pretrained=False, num_classes=10)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


train_loader, test_loader = datasets.mnist(download=False)

if args.cuda:
    cnn.cuda()

optimizer = optim.Adam(cnn.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    criterion.cuda()

history = model_fit(cnn, train_loader, criterion, optimizer, epochs=1, validation=test_loader, cuda=args.cuda)
