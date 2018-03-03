from torch.autograd import Variable

import time

def model_fit(model, train_loader, criterion, optimizer, epochs=1, validation=None, cuda=False):
    if validation is not None:
        print("Train on {} samples, validate on {} samples".format(
                    len(train_loader.dataset), len(validation.dataset)))
    else:
        print("Train on {} samples, no validate samples".format(
                    len(train_loader.dataset)))

    if cuda:
        print("Cuda: True")
    else:
        print("Cuda: False")

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
            total_correct += target.eq(output.max(dim=1)[1]).sum().data[0]

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        toc = time.time()

        if validation is None:
            print("Epoch {}/{} - {}s - loss: {} - acc: {}".format(
                epoch, epochs, toc-tic, total_loss/len(train_loader.dataset),
                total_correct/len(train_loader.dataset)))
            continue
        else:
            print("Epoch {}/{} - train - {}s - loss: {} - acc: {}".format(
                epoch, epochs, toc-tic, total_loss/len(train_loader.dataset),
                total_correct/len(train_loader.dataset)),
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
            total_correct += target.eq(output.max(dim=1)[1]).sum().data[0]
        toc = time.time()

        # test_loss /= len(validation.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(validation.dataset),
        #     100. * correct / len(validation.dataset)))
        print("- validate - {}s - val_loss: {} - val_acc: {}".format(
                toc-tic, total_loss/len(validation.dataset),
                total_correct/len(validation.dataset)))


