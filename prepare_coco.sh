#!/bin/bash
echo Download TorchVision ImageNet weights and the exact COCO weights from Hung et al.
wget http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth
python convert_coco_resnet101.py
echo 8 Unmatched weights on Cityscapes is not a problem.
echo Now you can conduct the experiments!
