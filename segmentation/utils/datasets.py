import torchvision
import os
import torch
import numpy as np
from PIL import Image
from utils.common import train_cities, label_id_map_city


# Reimplemented based on torchvision.datasets.VOCSegmentation to support unlabeled data
# Only file names are loaded in memory => On-the-fly pseudo labeling should be fine
# label_state = 0: ground truth
#               1: pseudo
#               2: no label
class StandardSegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None, label_state=0,
                 data_set='voc', mask_type='.png'):
        super().__init__(root, transforms, transform, target_transform)
        self.mask_type = mask_type
        if data_set == 'voc':
            self._voc_init(root, image_set, label_state)
        elif data_set == 'city':
            self._city_init(root, image_set, label_state)
        else:
            raise ValueError

        assert (len(self.images) == len(self.masks))

        # Different label states
        self.gt = (label_state == 0)
        if label_state == 2:
            self.has_label = False
        else:
            self.has_label = True

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.has_label:
            # Return x (input image) & y (mask images as a list)
            # Supports .png & .npy
            target = Image.open(self.masks[index]) if '.png' in self.masks[index] else np.load(self.masks[index])
            # Transforms
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target
        else:
            # Return x (input image) & filenames & original image size as a list to store pseudo label
            target = self.masks[index]
            w, h = img.size
            # Transforms
            if self.transforms is not None:
                img, target, h, w = self.transforms(img, target, h, w)
            return img, target, h, w

    def __len__(self):
        return len(self.images)

    def _voc_init(self, root, image_set, label_state):
        image_dir = os.path.join(root, 'JPEGImages')
        if label_state == 0:
            mask_dir = os.path.join(root, 'SegmentationClassAug')
        else:
            mask_dir = os.path.join(root, 'SegmentationClassAugPseudo')
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)

        splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]

    def _city_init(self, root, image_set, label_state):
        # It's tricky, lots of folders
        image_dir = os.path.join(root, 'leftImg8bit')
        check_dirs = False
        if label_state == 0:
            mask_dir = os.path.join(root, 'gtFine')
        else:
            mask_dir = os.path.join(root, 'gtFinePseudo')
            check_dirs = True

        if image_set == 'val' or image_set == 'test':
            image_dir = os.path.join(image_dir, image_set)
            mask_dir = os.path.join(mask_dir, image_set)
        elif image_set == 'valtiny':
            image_dir = os.path.join(image_dir, 'val')
            mask_dir = os.path.join(mask_dir, 'val')
        else:
            image_dir = os.path.join(image_dir, 'train')
            mask_dir = os.path.join(mask_dir, 'train')

        if check_dirs:
            for city in train_cities:
                temp = os.path.join(mask_dir, city)
                if not os.path.exists(temp):
                    os.makedirs(temp)

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, 'data_lists')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + "_leftImg8bit.png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_gtFine_labelIds" + self.mask_type) for x in file_names]


# With a sole purpose to elegantly & approximately evaluate pseudo labeling performance
class SegmentationLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set, data_set='voc', mask_type='.png'):
        if data_set == 'voc':
            pseudo_dir = os.path.join(root, 'SegmentationClassAugPseudo')
            target_dir = os.path.join(root, 'SegmentationClassAug')
            list_dir = os.path.join(root, 'ImageSets/Segmentation')
        elif data_set == 'city':
            pseudo_dir = os.path.join(root, 'gtFinePseudo/train')
            target_dir = os.path.join(root, 'gtFine/train')
            list_dir = os.path.join(root, 'data_lists')
        else:
            raise ValueError

        list_f = os.path.join(list_dir, image_set + '.txt')
        with open(os.path.join(list_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if data_set == 'voc':
            self.map = False
            self.pseudos = [os.path.join(pseudo_dir, x + mask_type) for x in file_names]
            self.targets = [os.path.join(target_dir, x + ".png") for x in file_names]
        else:
            self.map = True
            self.pseudos = [os.path.join(pseudo_dir, x + "_gtFine_labelIds" + mask_type) for x in file_names]
            self.targets = [os.path.join(target_dir, x + "_gtFine_labelIds.png") for x in file_names]

    def __getitem__(self, index):
        target = Image.open(self.targets[index])
        if '.png' in self.pseudos[index]:
            pseudo = Image.open(self.pseudos[index])
            pseudo = torch.as_tensor(np.asarray(pseudo), dtype=torch.int64)
        else:
            pseudo = np.load(self.pseudos[index])
            if pseudo.dim() == 3:  # Offline weights
                pseudo = torch.as_tensor(pseudo)[:, :, 0].long()
            else:
                pseudo = torch.as_tensor(pseudo).long()

        # Match pseudo label's shape
        target = torch.as_tensor(np.asarray(target), dtype=torch.float)
        target = target[None][None]
        target = torch.nn.functional.interpolate(target, size=(pseudo.shape[0], pseudo.shape[1]), mode='nearest')
        target = target[0][0].long()

        if self.map:
            target = torch.tensor(label_id_map_city)[target]

        return pseudo, target

    def __len__(self):
        return len(self.pseudos)
