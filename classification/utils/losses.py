# Implemented upon pytorch 1.2.0
import torch
import math


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
# Actually there is no need for ignore_index, just keep it to be consistent
class DynamicMutualLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, gamma1, gamma2, split_index=None, margin=0):
        # < split_index are pseudo labeled samples
        # Cross-entropy loss for all samples
        real_targets = targets.argmax(1).clone().detach()
        total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
        true_loss = total_loss.mean().item()
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0}

        if split_index is None or (gamma1 == 0 and gamma2 == 0):  # No dynamic loss
            total_loss = total_loss.sum() / (targets != self.ignore_index).sum()
        else:  # Dynamic loss
            outputs = inputs.softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)  # N
            decision_pseudo = real_targets.clone().detach()  # N
            confidence_current = outputs.max(1).values  # N
            confidence_pseudo = targets.max(1).values.clone().detach()  # N

            temp = decision_pseudo.unsqueeze(1).clone().detach()
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

            dynamic_weights = torch.ones_like(decision_current).float()

            # Prepare indices
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > (confidence_pseudo + margin)

            stats['disagree'] = disagreement[:split_index].sum().int().item()
            stats['current_win'] = (disagreement * current_win)[:split_index].sum().int().item()

            # Agree
            indices = ~disagreement
            dynamic_weights[indices] = probabilities_current[indices] ** gamma1

            # Disagree (current model wins, do not learn!)
            indices = disagreement * current_win
            dynamic_weights[indices] = 0

            # Disagree
            indices = disagreement * ~current_win
            dynamic_weights[indices] = probabilities_current[indices] ** gamma2

            # Weight loss
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights.mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()

        return total_loss, true_loss, stats


# Dynamic loss for DMT with mixup
class MixupDynamicLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, y_a, y_b, lam, w_a=None, w_b=None, dynamic_weights=None):
        # Weighted Cross-entropy loss for all samples
        if dynamic_weights is None:
            loss_a = self.criterion_ce(pred, y_a)
            loss_b = self.criterion_ce(pred, y_b)
            true_loss = (lam * loss_a + (1 - lam) * loss_b).mean().item()
            total_loss = lam * w_a * loss_a + (1 - lam) * w_b * loss_b
            total_loss = total_loss.sum() / ((y_a != self.ignore_index) * (y_b != self.ignore_index)).sum()
        else:
            total_loss = lam * self.criterion_ce(pred, y_a) + (1 - lam) * self.criterion_ce(pred, y_b)
            true_loss = total_loss.mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / \
                         ((y_a != self.ignore_index) * (y_b != self.ignore_index)).sum()

        return total_loss, true_loss

    @staticmethod  # TODO: Change to wrapper
    def dynamic_weights_calc(net, inputs, targets, split_index, gamma=0, labeled_weight=1):
        # Dynamic weights for whole batch (unlabeled data | split_index | labeled data)
        # For labeled data, weights are always 1
        with torch.no_grad():
            outputs = net(inputs).softmax(dim=1).clone().detach()
            indices = targets.unsqueeze(1).clone().detach()
            probabilities = outputs.gather(dim=1, index=indices).squeeze(1)
            probabilities[split_index:] = labeled_weight
            probabilities[:split_index] = probabilities[:split_index] ** gamma

        return probabilities


# Dynamic loss for DMT (Dynamic Mutual-Training) with mixup
class SigmoidAscendingMixupDMTLoss(MixupDynamicLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1, gamma2, T_max, weight=None, ignore_index=-100, reduction='mean'):
        super(MixupDynamicLoss, self).__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.gamma1_ori = gamma1
        self.gamma2_ori = gamma2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.T_max = T_max
        self.last_iter = 1

    def step(self):
        self.last_iter += 1
        ratio = math.e ** (-5 * (1 - self.last_iter / self.T_max) ** 2)
        self.gamma1 = self.gamma1_ori * ratio
        self.gamma2 = self.gamma2_ori * ratio

    def dynamic_weights_calc(self, net, inputs, targets, split_index, labeled_weight=1, margin=0):
        # Dynamic weights for whole batch (unlabeled data | split_index | labeled data)
        # For labeled data, weights are always 1
        # targets: Original softmax results for pseudo labels
        # inputs: Input images
        # gamma1: For agreement loss
        # gamma2: For disagreement loss
        # margin: Current model has to win by a sufficient margin
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, "gamma1": self.gamma1, "gamma2": self.gamma2}
        with torch.no_grad():
            outputs = net(inputs).softmax(dim=1).clone().detach()  # N * num_classes
            decision_current = outputs.argmax(1)  # N
            decision_pseudo = targets.argmax(1).clone().detach()  # N
            confidence_current = outputs.max(1).values  # N
            confidence_pseudo = targets.max(1).values.clone().detach()  # N

            temp = decision_pseudo.unsqueeze(1).clone().detach()
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

            dynamic_weights = torch.ones_like(decision_current).float()

            # Prepare indices
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = disagreement[:split_index].sum().int().item()
            stats['current_win'] = (disagreement * current_win)[:split_index].sum().int().item()
            if self.gamma1 == self.gamma2 == 0:  # Temp patch
                return torch.ones(inputs.shape[0]).to(inputs.device), stats

            # Agree
            indices = ~disagreement
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1

            # Disagree (current model wins, do not learn!)
            indices = disagreement * current_win
            dynamic_weights[indices] = 0

            # Disagree
            indices = disagreement * ~current_win
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2

            # Labeled portion
            dynamic_weights[split_index:] = labeled_weight
            stats['avg_weights'] = dynamic_weights.mean().item()

        return dynamic_weights, stats
