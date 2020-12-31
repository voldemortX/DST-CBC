# Modified from ildoonet/pytorch-randaugment
from RandAugment.augmentations import augment_list
import random


# RandAugment with random magnitude (1-10)
# Cutout included
class RandomRandAugment:
    def __init__(self, n, m_max=10):
        self.n = n
        self.m_max = m_max  # [1, 10]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        m = random.randint(1, self.m_max)
        for op, min_val, max_val in ops:
            val = (float(m) / 10) * float(max_val - min_val) + min_val
            img = op(img, val)

        return img
