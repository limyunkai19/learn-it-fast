import torch
from torch.autograd import Variable

import os
import json
import time

class History:
    def __init__(self):
        self.meta = {}
        self.epoch_history = {}
        self.iter_history = {}

    def get_dict(self):
        hist =  {
                    'meta': self.meta,
                    'epoch_history': self.epoch_history,
                    'iter_history': self.iter_history
                }
        return hist

    def from_dict(self, hist):
        self.meta = hist['meta']
        self.epoch_history = hist['epoch_history']
        self.iter_history = hist['iter_history']

        return self

    # def visualize(self, iter=False):


def model_fit(model, train_loader, criterion, optimizer, epochs=1, validation=None, cuda=False, save_best_name=None, verbose=2):
    history = History()

    if validation is not None:
        print("Train on {} samples, validate on {} samples".format(
                    len(train_loader.sampler), len(validation.sampler)))
        history.meta['train_size'] = len(train_loader.sampler)
        history.meta['val_size'] = len(validation.sampler)
        if 'best_val_acc' not in model.meta:
            model.meta['best_val_acc'] = -1
    else:
        print("Train on {} samples, no validate samples".format(
                    len(train_loader.sampler)))
        history.meta['train_size'] = len(train_loader.sampler)
        save_best_name = None

    print("Cuda: {}".format(cuda))

    history.meta['cuda'] = cuda
    history.meta['epoch'] = epochs
    history.epoch_history['loss'] = []
    history.epoch_history['acc'] = []
    history.iter_history['loss'] = []
    history.iter_history['acc'] = []

    if validation is not None:
        history.epoch_history['val_loss'] = []
        history.epoch_history['val_acc'] = []
        history.iter_history['val_loss'] = []
        history.iter_history['val_acc'] = []

    ### begin epoch loop
    for epoch in range(1, epochs+1):

        ### train
        tic = time.time()
        model.train()
        total_loss = 0
        total_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # forward + backward
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward(), optimizer.step()

            # log stats
            total_loss += loss.data[0]*len(target)
            correct = target.eq(output.max(dim=1)[1]).short().sum().data[0]
            total_correct += correct
            history.iter_history['loss'].append(loss.data[0])
            history.iter_history['acc'].append(correct/len(target))

            if verbose >= 2:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        toc = time.time()

        loss = total_loss/len(train_loader.sampler)
        acc = total_correct/len(train_loader.sampler)
        history.epoch_history['loss'].append(loss)
        history.epoch_history['acc'].append(acc)
        if validation is None:
            print("Epoch {}/{} - {}s - loss: {} - acc: {}".format(
                epoch, epochs, toc-tic, loss, acc))
            continue
        else:
            print("Epoch {}/{} - train - {}s - loss: {} - acc: {}".format(
                epoch, epochs, toc-tic, loss, acc),
                end=" ")

        ### test
        tic = time.time()
        model.eval()
        total_loss = 0
        total_correct = 0
        for data, target in validation:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            # forward
            output = model(data)
            loss = criterion(output, target)

            # log stats
            total_loss += loss.data[0]*len(target)
            correct = target.eq(output.max(dim=1)[1]).short().sum().data[0]
            total_correct += correct
            history.iter_history['val_loss'].append(loss.data[0])
            history.iter_history['val_acc'].append(correct/len(target))
        toc = time.time()

        loss = total_loss/len(validation.sampler)
        acc = total_correct/len(validation.sampler)
        history.epoch_history['val_loss'].append(loss)
        history.epoch_history['val_acc'].append(acc)
        print("- validate - {}s - val_loss: {} - val_acc: {}".format(
                toc-tic, loss, acc))

        if save_best_name is not None and acc > model.meta['best_val_acc']:
            model.meta['best_val_acc'] = acc
            model_checkpoint(model, save_best_name, 'best_model')

    return history

def model_eval(model, test_loader, criterion, cuda=False, verbose=True):
    tic = time.time()
    model.eval()
    total_loss = 0
    total_correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # forward
        output = model(data)
        loss = criterion(output, target)

        # log stats
        total_loss += loss.data[0]*len(target)
        correct = target.eq(output.max(dim=1)[1]).short().sum().data[0]
        total_correct += correct
    toc = time.time()

    loss = total_loss/len(test_loader.sampler)
    acc = total_correct/len(test_loader.sampler)
    print("Test - {}s - loss: {} - acc: {}/{} {}".format(
            toc-tic, loss, total_correct, len(test_loader.sampler), acc))

def model_checkpoint(model, name, checkpoint_name, base_path='results'):
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    working_dir = os.sep.join([base_path, name])
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    torch.save(model.state_dict(), os.sep.join([working_dir, '{}.pth'.format(checkpoint_name)]))


def model_save(model, history, name, base_path='results', save_state=True):
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    working_dir = os.sep.join([base_path, name])
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    if save_state:
        torch.save(model.state_dict(), os.sep.join([working_dir, 'model.pth']))

    with open(os.sep.join([working_dir, 'meta.json']), 'w') as f:
        json.dump(model.meta, f)

    with open(os.sep.join([working_dir, 'history.json']), 'w') as f:
        json.dump(history.get_dict(), f)

    return working_dir

def model_load(name, base_path='results'):
    working_dir = os.sep.join([base_path, name])
    if not os.path.isdir(working_dir):
        print("Saves not found")
        return None

    with open(os.sep.join([working_dir, 'meta.json']), 'r') as f:
        meta = json.load(f)

    import models
    model = models.__dict__[meta['base_model']](
        num_classes=meta['num_classes'],
        pretrained=meta['pretrained'],
        mode=meta['mode']
    )
    model.load_state_dict(torch.load(os.sep.join([working_dir, 'model.pth'])))

    return model

def prod(lists):
    ans = 1
    for i in lists:
        ans *= i
    return ans

def num_param(model):
    ans = 0
    for param in model.parameters():
        ans += prod(param.size())

    return ans


def num_trainable_param(model):
    ans = 0
    for param in model.parameters():
        if param.requires_grad:
            ans += prod(param.size())

    return ans

def get_trainable_param(model):
    return filter(lambda p: p.requires_grad, model.parameters())
