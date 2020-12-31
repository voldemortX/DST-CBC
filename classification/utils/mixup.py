# Copied from facebookresearch/mixup-cifar10
import torch
import numpy as np


def mixup_data(x, y, alpha=1.0, keep_max=False, w=None):
    # Returns mixed inputs, pairs of targets, and lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if keep_max:  # For semi-supervised learning, identity needs keeping
        lam = max(lam, 1 - lam)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    if w is not None:  # For DMT, where corresponding weights also need to be mixed
        mixed_w = lam * w + (1 - lam) * w[index]

    y_a, y_b = y, y[index]

    if w is None:
        return mixed_x, y_a, y_b, lam
    else:
        return mixed_x, mixed_w, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
