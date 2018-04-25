import os
from .folder import folder


def folder2(root, train_dir='train', test_dir='val', data_augmentation=False, sample_per_class=(-1, -1), **kwargs):
    train_data = os.path.join(root, train_dir)
    train_loader = folder(train_data, data_augmentation=data_augmentation, sample_per_class=sample_per_class[0], **kwargs)

    test_data = os.path.join(root, test_dir)
    test_loader = folder(test_data, data_augmentation=False, sample_per_class=sample_per_class[1], **kwargs)


    return train_loader, test_loader
