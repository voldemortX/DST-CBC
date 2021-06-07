import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
from accelerate import Accelerator
from torch.cuda.amp import autocast


mean_cifar10 = [0.49137, 0.48235, 0.44667]
std_cifar10 = [0.24706, 0.24353, 0.26157]
num_classes_cifar10 = 10
input_sizes_cifar10 = (32, 32)
base_cifar10 = '../../../dataset/cifar10'


# Draw images/labels from tensors
def show(images, std, mean):
    np_images = images.numpy()
    # Denormalize and map from (N, 3, d1, d2) to (N, d1, d2, 3)
    np_images = np.transpose(np_images, (0, 2, 3, 1))
    np_images = np_images * std + mean
    if mean[0] > 1:
        np_images /= 255.0

    plt.imshow(np_images.reshape((np_images.shape[0] * np_images.shape[1], np_images.shape[2], np_images.shape[3])))
    plt.show()


# Save model checkpoints(supports amp)
def save_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename='temp.pt'):
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'amp': amp.state_dict() if is_mixed_precision else None
    }
    torch.save(checkpoint, filename)


# Load model checkpoints(supports amp)
def load_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if is_mixed_precision and checkpoint['amp'] is not None:
        amp.load_state_dict(checkpoint['amp'])


# Count for threshold (k) to select top confident labels
def rank_label_confidence(net, device, loader, ratio, num_images, is_mixed_precision):
    net.eval()
    if ratio >= 1:
        k = 0
    else:
        # 1 forward pass (record predicted probabilities)
        probabilities = np.ones(num_images, dtype=np.float32)
        i = 0
        with torch.no_grad():
            for images, _ in tqdm(loader):
                # Inference
                images = images.to(device)
                with autocast(is_mixed_precision):
                    outputs = net(images)
                    temp = torch.nn.functional.softmax(input=outputs, dim=1)  # ! softmax
                    pseudo_probabilities = temp.max(dim=1).values
                temp_len = pseudo_probabilities.shape[0]

                # Count
                probabilities[i: i + temp_len] = pseudo_probabilities.cpu().numpy()
                i += temp_len

        # Sort(n * log(n) << n * n * label_ratio, so just sort is good) and find k
        print('Sorting...')
        probabilities.sort()
        k = probabilities[int(-ratio * num_images + 1)]
        print('Done.')

    return k


# Keep a moving average of model weights in GPU
# Not running mean and variance in BN
class EMA(object):
    def __init__(self, net, decay):
        self.shadows = collections.OrderedDict()
        self.decay = decay
        for name, param in net.named_parameters():
            self.shadows[name] = param.data.clone().detach()

    def update(self, net):
        for name, param in net.named_parameters():
            # self.shadows[name] = (1.0 - self.decay) * self.shadows[name] + self.decay * param.data.clone().detach()
            # For lockless ops, although mostly not needed
            # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            self.shadows[name] -= self.decay * (self.shadows[name] - param.data.clone().detach())

    def fill_in_bn(self, state_dict):
        for key in state_dict.keys():
            if ('running_mean' in key or 'running_var' in key) and key not in self.shadows.keys():
                self.shadows[key] = state_dict[key].clone()

    def state_dict(self):
        return self.shadows
