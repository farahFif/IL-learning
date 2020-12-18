import numpy as np
import  torch
from torchvision import datasets, transforms


class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name, labels_per_class_train, labels_per_class_test):
        self.classes = classes
        self.name = name
        self.train_data = None
        self.test_data = None
        self.labels_per_class_train = labels_per_class_train
        self.labels_per_class_test = labels_per_class_test
        self.nb_class = 10
        self.labels = None
        self.labels_test = None


class CIFAR10(Dataset):
    def __init__(self):
        nb_class = 10
        super().__init__(nb_class, "CIFAR10", 5000, 1000)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), ])

        self.train_data = datasets.CIFAR10("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR10("data", train=False, transform=self.test_transform, download=True)

        self.labels = np.asarray(self.train_data.targets)
        self.labels_test = np.asarray(self.test_data.targets)
        self.transformLabels()

        self.nb_class = nb_class

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b
        # change labels for test
        c = np.zeros((self.labels_test.size, self.labels_test.max() + 1))
        c[np.arange(self.labels_test.size), self.labels_test] = 1
        self.labels_test = c
        self.unique_labels = np.unique(c, axis=0)

class CIFAR100(Dataset):
    def __init__(self):
        nb_class = 100
        super().__init__(nb_class, "CIFAR100", 500, 100)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), ])

        self.train_data = datasets.CIFAR100("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR100("data", train=False, transform=self.test_transform, download=True)

        self.labels = np.asarray(self.train_data.targets)
        self.labels_test = np.asarray(self.test_data.targets)
        self.transformLabels()

        self.nb_class = nb_class


    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b
        # change labels for test
        c = np.zeros((self.labels_test.size, self.labels_test.max() + 1))
        c[np.arange(self.labels_test.size), self.labels_test] = 1
        self.labels_test = c
        self.unique_labels = np.unique(c, axis=0)