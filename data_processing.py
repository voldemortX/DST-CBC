import torchvision
import os
import torch
import numpy as np
from PIL import Image

# Base directories
base_voc = '../voc_seg_deeplab/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
base_city = '../../dataset/cityscapes'

# Common parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
coco_mean = [104.008, 116.669, 122.675]  # BGR
coco_std = [1.0, 1.0, 1.0]
city_mean = [73.15835918458554, 82.90891773640608, 72.39239908619095]
voc_mean = [116.52101153914718, 111.3575037556515, 102.92616541705553]

# Here 'training resize min' is also the final training crop size as RandomResize & RandomCrop are used together
# For PASCAL VOC 2012
sizes_voc = [(321, 321), (505, 505), (505, 505)]  # training resize min/training resize max/testing label size
num_classes_voc = 21
colors_voc = [[0, 0, 0],
              [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [255, 255, 255]]
categories_voc = ['Background',
                  'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                  'Bottle', 'Bus', 'Car', 'Cat',
                  'Chair', 'Cow', 'Diningtable', 'Dog',
                  'Horse', 'Motorbike', 'Person', 'Pottedplant',
                  'Sheep', 'Sofa', 'Train', 'Tvmonitor']

# For cityscapes (19 classes, ignore as black, no such thing as background)
sizes_city = [(256, 512), (512, 1024), (512, 1024)]  # training resize min/training resize max/testing label size
num_classes_city = 19
colors_city = [
               [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32],
               [0, 0, 0]]
categories_city = [
                   'road', 'sidewalk', 'building', 'wall',
                   'fence', 'pole', 'traffic light', 'traffic sign',
                   'vegetation', 'terrain', 'sky', 'person',
                   'rider', 'car', 'truck', 'bus',
                   'train', 'motorcycle', 'bicycle']
label_id_map_city = [255, 255, 255, 255, 255, 255, 255,
                     0,   1,   255, 255, 2,   3,   4,
                     255, 255, 255, 5,   255, 6,   7,
                     8,   9,   10,  11,  12,  13,  14,
                     15,  255, 255, 16,  17,  18]
train_cities = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover',
                'krefeld', 'strasbourg', 'tubingen', 'weimar', 'bochum',
                'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach',
                'stuttgart', 'ulm', 'zurich']


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
            # Return x(input image) & y(mask images as a list)
            # Supports .png & .npy
            target = Image.open(self.masks[index]) if '.png' in self.masks[index] else np.load(self.masks[index])
        else:
            # Return x(input image) & file names as a list to store pseudo label
            target = self.masks[index]

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

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
