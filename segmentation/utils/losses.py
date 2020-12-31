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


# Dynamic loss for DMT (Dynamic Mutual Training): Weights depend on confidence
class DynamicLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        statistics = {}
        total_loss = self.criterion_ce(inputs, targets)  # Pixel-level weight is always 1

        if split_index is None or self.gamma == 0:  # No dynamic loss
            total_loss = total_loss.sum() / (targets != self.ignore_index).sum()
        else:  # Dynamic loss
            probabilities = inputs.softmax(dim=1).clone().detach()
            indices = targets.unsqueeze(1).clone().detach()
            indices[indices == self.ignore_index] = 0
            probabilities = probabilities.gather(dim=1, index=indices).squeeze(1)
            probabilities[split_index:] = 1
            probabilities[:split_index] = probabilities[:split_index] ** self.gamma
            total_loss = (total_loss * probabilities).sum() / (targets != self.ignore_index).sum()

        statistics['dl'] = total_loss.item()
        return total_loss, statistics


# Dynamic loss for DMT (Dynamic Mutual-Training): Weights depend on confidence
class DynamicMutualLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1=0, gamma2=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        # Targets could be soft (for DMT) or hard (for fully-supervised training)
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets
        total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0,
                 'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}

        if split_index is None or (self.gamma1 == 0 and self.gamma2 == 0):  # No dynamic loss
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:  # Dynamic loss
            outputs = inputs.softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)  # N
            decision_pseudo = real_targets.clone().detach()  # N
            confidence_current = outputs.max(1).values  # N

            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

            confidence_pseudo = targets[:, 1, :, :].clone().detach()  # N
            dynamic_weights = torch.ones_like(decision_current).float()

            # Prepare indices
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = (disagreement * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            stats['current_win'] = ((disagreement * current_win) * (real_targets != self.ignore_index))[:split_index] \
                .sum().int().item()

            # Agree
            indices = ~disagreement
            # dynamic_weights[indices] = (confidence_current[indices] * confidence_pseudo[indices]) ** self.gamma1
            # dynamic_weights[indices] = (probabilities_current[indices] ** self.gamma2) * (confidence_pseudo[indices] ** self.gamma1)
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1

            # Disagree (current model wins, do not learn!)
            indices = disagreement * current_win
            dynamic_weights[indices] = 0

            # Disagree
            indices = disagreement * ~current_win
            # dynamic_weights[indices] = ((1 - confidence_current[indices]) * confidence_pseudo[indices]) ** self.gamma2
            # dynamic_weights[indices] = (probabilities_current[indices] ** self.gamma2) * (confidence_pseudo[indices] ** self.gamma1)
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2

            # Weight loss
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()

        return total_loss, stats


# Naive dynamic loss same as DST-CBC
class DynamicNaiveLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        # Targets could be soft (for DMT) or hard (for fully-supervised training)
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets
        total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0,
                 'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}

        if split_index is None:  # No dynamic loss
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:  # Dynamic loss
            outputs = inputs.softmax(dim=1).clone().detach()
            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)
            dynamic_weights = probabilities_current ** 5

            # Weight loss
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()

        return total_loss, stats


# For ablations
class FlipLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1=0, gamma2=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        # Targets could be soft (for DMT) or hard (for fully-supervised training)
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets

        if split_index is None or (self.gamma1 == 0 and self.gamma2 == 0):  # No dynamic loss
            total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
            stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0,
                     'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:  # Dynamic loss
            stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0,
                     'loss': 0 / (real_targets != self.ignore_index).sum().item()}
            outputs = inputs.softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)  # N
            decision_pseudo = real_targets.clone().detach()  # N
            confidence_current = outputs.max(1).values  # N

            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

            confidence_pseudo = targets[:, 1, :, :].clone().detach()  # N
            dynamic_weights = torch.ones_like(decision_current).float()

            # Prepare indices
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = (disagreement * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            stats['current_win'] = ((disagreement * current_win) * (real_targets != self.ignore_index))[:split_index] \
                .sum().int().item()

            # Agree
            indices = ~disagreement
            # dynamic_weights[indices] = (confidence_current[indices] * confidence_pseudo[indices]) ** self.gamma1
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1

            # Disagree (current model wins, flip!)
            indices = disagreement * current_win * (real_targets != self.ignore_index)
            dynamic_weights[indices] = (1 - confidence_pseudo[indices]) ** self.gamma2
            real_targets[:split_index][indices[:split_index]] = decision_current[:split_index][indices[:split_index]]

            # Disagree
            indices = disagreement * ~current_win
            # dynamic_weights[indices] = ((1 - confidence_current[indices]) * confidence_pseudo[indices]) ** self.gamma2
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2

            # Weight loss
            total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
            stats['loss'] = total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()

        return total_loss, stats


class OnlineLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, inputs, targets, split_index=None):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples

        outputs = inputs.softmax(dim=1).clone().detach()
        pseudo_targets = outputs.argmax(1)
        confidences = outputs.max(1).values
        pseudo_targets[confidences <= 0.9] = 255
        targets[:split_index] = pseudo_targets[:split_index]
        total_loss = self.criterion_ce(inputs, targets)
        stats = {'loss': total_loss.item()}

        return total_loss, stats
