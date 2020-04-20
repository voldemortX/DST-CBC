# Implemented upon pytorch 1.2.0
import torch


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, *input):
        raise NotImplementedError


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)

    def forward(self, *input):
        raise NotImplementedError


# Dynamic loss for DST (Dynamic Self-Training): Weights depend on confidence
class DynamicLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, original_conf=None, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        statistics = {}
        total_loss = self.criterion_ce(inputs, targets)  # Pixel-level weight is always 1

        if split_index is None or self.gamma == 0:  # No dynamic loss
            total_loss = total_loss.sum() / (targets != self.ignore_index).sum()
        else:  # Dynamic loss
            probabilities = torch.ones_like(total_loss)
            probabilities[:split_index] = original_conf
            probabilities[split_index:] = 1
            probabilities[:split_index] = probabilities[:split_index] ** self.gamma
            total_loss = (total_loss * probabilities).sum() / (targets != self.ignore_index).sum()

        statistics['dl'] = total_loss.item()
        return total_loss, statistics
