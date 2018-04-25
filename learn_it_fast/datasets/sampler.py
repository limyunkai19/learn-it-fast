import torch
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler

class BalancedSubsetRandomSampler(SubsetRandomSampler):
    """Samples a subset of data and ensure that each class has same number of element.

    Arguments:
        data_source (Dataset): dataset to sample from
        sample_per_class (int): number of element per class to sample
        num_classes (int):total number of class exist in dataset
    """
    def __init__(self, data_source, sample_per_class, num_classes):
        self.indices = self.get_indices(data_source, sample_per_class, num_classes)

    def get_indices(self, data_source, sample_per_class, num_classes):
        sampled = 0
        indices = []
        sampled_class = [0]*num_classes
        total_to_sampled = sample_per_class*num_classes

        for i in torch.randperm(len(data_source)):
            x, y = data_source[i]

            if sampled_class[y] < sample_per_class:
                sampled += 1
                indices.append(i)
                sampled_class[y] += 1

            if sampled >= total_to_sampled:
                break

        return indices
