from torch.autograd import Variable

import time

class History:
    def __init__(self):
        self.meta = {}
        self.epoch_history = {}
        self.iter_history = {}

def model_fit(model, train_loader, criterion, optimizer, epochs=1, validation=None, cuda=False):
    history = History()

    if validation is not None:
        print("Train on {} samples, validate on {} samples".format(
                    len(train_loader.dataset), len(validation.dataset)))
        history.meta['train_size'] = len(train_loader.dataset)
        history.meta['val_size'] = len(validation.dataset)
    else:
        print("Train on {} samples, no validate samples".format(
                    len(train_loader.dataset)))
        history.meta['train_size'] = len(train_loader.dataset)

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
            correct = target.eq(output.max(dim=1)[1]).sum().data[0]
            total_correct += correct
            history.iter_history['loss'].append(loss.data[0])
            history.iter_history['acc'].append(correct/len(target))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        toc = time.time()

        loss = total_loss/len(train_loader.dataset)
        acc = total_correct/len(train_loader.dataset)
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
            correct = target.eq(output.max(dim=1)[1]).sum().data[0]
            total_correct += correct
            history.iter_history['val_loss'].append(loss.data[0])
            history.iter_history['val_acc'].append(correct/len(target))
        toc = time.time()

        # test_loss /= len(validation.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(validation.dataset),
        #     100. * correct / len(validation.dataset)))

        loss = total_loss/len(validation.dataset)
        acc = total_correct/len(validation.dataset)
        history.epoch_history['val_loss'].append(loss)
        history.epoch_history['val_acc'].append(acc)
        print("- validate - {}s - val_loss: {} - val_acc: {}".format(
                toc-tic, loss, acc))

    return history
