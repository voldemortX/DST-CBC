import torch
from models.segmentation.segmentation import deeplab_v2
from utils.common import save_checkpoint

# COCO pre-trained Deeplab-ResNet-101 from Hung et al.
# (checked to be the same as the original caffe model published by the DeeplabV2 authors)
# (_-_) Gave them a star on github to show our gratitude
# http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth
# This script matches this pre-trained model's parameter dict keys with our implementation,
# We have 104 extra .num_batches_tracked, which are all tensor(0), others are the same
hung_coco_filename = 'resnet101COCO-41f33a49.pth'
coco = torch.load(hung_coco_filename)
voc_net = deeplab_v2(num_classes=21)
city_net = deeplab_v2(num_classes=19)
my_voc = voc_net.state_dict().copy()
my_city = city_net.state_dict().copy()
voc_shape_not_match = 0
voc_shape_match = 0
city_shape_not_match = 0
city_shape_match = 0

for key in coco:
    if 'layer5' in key:
        my_key = 'classifier.0.convs' + key.split('conv2d_list')[1]
    else:
        my_key = 'backbone.' + key
    if my_voc[my_key].shape == coco[key].shape:
        voc_shape_match += 1
        my_voc[my_key] = coco[key]
    else:
        voc_shape_not_match += 1
    if my_city[my_key].shape == coco[key].shape:
        city_shape_match += 1
        my_city[my_key] = coco[key]
    else:
        city_shape_not_match += 1

print(str(voc_shape_match) + ' pascal voc shapes matched!')
print(str(voc_shape_not_match) + ' pascal voc shapes are not a match.')
print(str(city_shape_match) + ' cityscapes shapes matched!')
print(str(city_shape_not_match) + ' cityscapes shapes are not a match.')
print('Saving models...')

voc_net.load_state_dict(my_voc)
city_net.load_state_dict(my_city)
save_checkpoint(net=voc_net, optimizer=None, lr_scheduler=None, is_mixed_precision=False,
                filename='voc_coco_resnet101.pt')
save_checkpoint(net=city_net, optimizer=None, lr_scheduler=None, is_mixed_precision=False,
                filename='city_coco_resnet101.pt')
print('Complete.')


# Outputs should be the following after a few seconds:
# 528 pascal voc shapes matched!
# 0 pascal voc shapes are not a match.
# 520 cityscapes shapes matched!
# 8 cityscapes shapes are not a match.
# Saving models...
# Complete.
