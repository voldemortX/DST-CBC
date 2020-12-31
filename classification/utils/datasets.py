# Modified from torchvision 0.4.0
from __future__ import print_function
from PIL import Image
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        set_name (char): Data set name, e.g. test, valtiny_seed1, trainval_seed1,
                         4_seed5_labeled, 25_seed1_unlabeled, etc.
        label (bool): If labels are needed
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'cifar-10-batches-py'

    def __init__(self, root, set_name, label=True, transform=None, target_transform=None):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        # Load files (load everything, since they are really small files)
        file_path = os.path.join(self.root, self.base_folder, set_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.targets = entry['labels'] if label is True else self.data.copy()  # Original np.ndarr
            else:
                self.targets = entry['fine_labels'] if label is True else self.data.copy()  # for CIFAR100

        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class when labeled;
            Or (image, corresponding original array) when unlabeled
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
