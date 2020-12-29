# https://github.com/galatolofederico/pytorch-balanced-batch

import torch

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    すべてのクラスからn_samples個選んで、それぞれのクラスから同じ数だけ取り出し1バッチ作る
    マイナークラスはover samplingする

    batch_sizeは必ずクラス数の整数倍を使用すること
    たとえば、10クラスならbatch_size=30

    Usage:
        import torch
        import pandas as pd

        epochs = 3
        size = 200
        features = 1
        classes_prob = torch.tensor([0.1, 0.3, 0.5, 0.01, 0.09])

        dataset_X = torch.randn(size, features)
        dataset_Y = torch.distributions.categorical.Categorical(classes_prob.repeat(size, 1)).sample()

        x_numpy = dataset_X.to('cpu').detach().numpy().copy()
        y_numpy = dataset_Y.to('cpu').detach().numpy().copy()
        #print(x_numpy.shape, x_numpy)
        print(y_numpy.shape, y_numpy)
        print(pd.Series(y_numpy).value_counts())

        dataset = torch.utils.data.TensorDataset(dataset_X, dataset_Y)

        batch_size = 10
        train_loader = torch.utils.data.DataLoader(dataset, sampler=BalancedBatchSampler(dataset, dataset_Y), batch_size=batch_size)

        for epoch in range(0, epochs):
            for batch_x, batch_y in train_loader:
                print("epoch: %d labels: %s\ninputs: %s\n" % (epoch, batch_y, batch_x))
    """

    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = (
                len(self.dataset[label])
                if len(self.dataset[label]) > self.balanced_max
                else self.balanced_max
            )

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][
                self.indices[self.currentkey]
            ]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif (
                is_torchvision_installed
                and dataset_type is torchvision.datasets.ImageFolder
            ):
                return dataset.imgs[idx][1]
            else:
                raise Exception(
                    "You should pass the tensor of labels to the constructor as second argument"
                )

    def __len__(self):
        return self.balanced_max * len(self.keys)
