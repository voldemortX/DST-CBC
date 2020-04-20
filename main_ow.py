# Recommend training a supervised baseline first,
# then conduct self-training from it to avoid mini-batch size issues

import os
import time
import copy
import torch
import argparse
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from all_utils_ow import init, deeplab_v2, train_one_iter, self_train_schedule, test_one_set, load_checkpoint
from data_processing import colors_city, colors_voc, categories_voc, categories_city, sizes_city, sizes_voc, \
                               num_classes_voc, num_classes_city, coco_mean, coco_std, imagenet_mean, imagenet_std
from losses_ow import DynamicLoss


torch.manual_seed(4396)
random.seed(7777)
np.random.seed(7777)
#torch.backends.cudnn.deterministic = True  # Might hurt performance
#torch.backends.cudnn.benchmark = False  # Might hurt performance


def after_loading():
    global lr_scheduler

    # The "poly" policy, variable names are confusing(May need reimplementation)
    if (args.state == 0) or (args.state == 1 and args.init_epochs == 0):
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda x: (1 - x / (len(labeled_loader) * args.init_epochs))
                                                         ** 0.9)
    # Resume training?
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)

    #visualize(labeled_loader)


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.2.0 && torchvision 0.4.0')
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city) (default: voc)')
    parser.add_argument('--gamma', type=float, default=2,
                        help='Gamma value for reverse focal loss (default: 2)')
    parser.add_argument('--val-num-steps', type=int, default=1000,
                        help='How many steps between validations (default: 1000)')
    parser.add_argument('--init-label-ratio', type=float, default=0.2,
                        help='Initial labeling ratio (default: 0.2)')
    parser.add_argument('--det-label-ratio', type=float, default=0.2,
                        help='Delta labeling ratio after each iteration (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.002)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--st-lr', type=float, default=0.002,
                        help='Initial learning rate for self-training (default: 0.002)')
    parser.add_argument('--init-epochs', type=int, default=30,
                        help='Number of epochs for the fully-supervised initialization (default: 30)')
    parser.add_argument('--st-epochs', type=int, nargs='+', default=[6, 6, 6, 6, 6],
                        help='Number of self-training epochs within each iteration (default: 66666)')
    parser.add_argument('--iters', type=int, default=5,
                        help='Number of self-training iterations (default: 5)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size (default: 8)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='Save model (default: True)')
    parser.add_argument('--coco', action='store_true', default=False,
                        help='Models started from COCO in Caffe(True) or ImageNet in Pytorch(False) (default: False)')
    parser.add_argument('--valtiny', action='store_true', default=False,
                        help='Use valtiny instead of val (default: False)')
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Turn off data augmentations for pseudo labeled data (default: False)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Self-training begins from a previous checkpoint/Test on this')
    parser.add_argument('--split', type=int, default=1,
                        help='e.g. 1:7(8), 1:3(4), 1:1(2), 1:0(1) labeled/unlabeled split (default: 1)')
    parser.add_argument('--sets-id', type=int, default=0,
                        help='Different random splits(0/1/2) (default: 0)')
    parser.add_argument('--state', type=int, default=1,
                        help="Final test(3)/Fully-supervised training(2)/Semi-supervised training(1)")
    args = parser.parse_args()

    # Basic configurations
    exp_name = str(int(time.time()))
    if args.exp_name != 'auto':
        exp_name = args.exp_name
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    if args.coco:  # This Caffe pre-trained model takes "inhuman" mean/std & input format
        mean = coco_mean
        std = coco_std
        keep_scale = True
        reverse_channels = True
    else:
        mean = imagenet_mean
        std = imagenet_std
        keep_scale = False
        reverse_channels = False
    if args.dataset == 'voc':
        num_classes = num_classes_voc
        input_sizes = sizes_voc
        categories = categories_voc
        colors = colors_voc
    elif args.dataset == 'city':
        num_classes = num_classes_city
        input_sizes = sizes_city
        categories = categories_city
        colors = colors_city
    else:
        raise ValueError

    net = deeplab_v2(num_classes=num_classes)
    print(device)
    net.to(device)

    # Define optimizer
    # Use different learning rates if you want, we do not observe improvement from different learning rates
    params_to_optimize = [
        {"params": [p for p in net.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in net.classifier.parameters() if p.requires_grad]},
    ]

    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.mixed_precision:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    # Just to be safe (a little bit more memory, by all means, save it to disk if you want)
    if args.state == 1:
        st_optimizer_init = copy.deepcopy(optimizer.state_dict())

    # Testing
    if args.state == 3:
        test_loader = init(batch_size=args.batch_size, state=3, split=None, valtiny=args.valtiny, no_aug=args.no_aug,
                           input_sizes=input_sizes, data_set=args.dataset, sets_id=args.sets_id,
                           mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        test_one_set(loader=test_loader, device=device, net=net, categories=categories, num_classes=num_classes,
                     output_size=input_sizes[2])
    else:
        x = 0
        criterion = DynamicLoss(gamma=args.gamma, ignore_index=255)
        writer = SummaryWriter('logs/' + exp_name)

        # Only fully-supervised training
        if args.state == 2:
            labeled_loader, val_loader = init(batch_size=args.batch_size, sets_id=args.sets_id, valtiny=args.valtiny,
                                              state=2, split=args.split, input_sizes=input_sizes, data_set=args.dataset,
                                              mean=mean, std=std, keep_scale=keep_scale, no_aug=args.no_aug,
                                              reverse_channels=reverse_channels)
            after_loading()
            x = train_one_iter(writer=writer, loader_c=labeled_loader, loader_sup=None, validation_loader=val_loader,
                               device=device, criterion=criterion, net=net, optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               num_epochs=args.init_epochs, categories=categories, num_classes=num_classes,
                               is_mixed_precision=args.mixed_precision, with_sup=False,
                               val_num_steps=args.val_num_steps, input_sizes=input_sizes)

        # Self-training
        elif args.state == 1:
            labeled_loader, pseudo_labeled_loader, unlabeled_loader, val_loader, reference_loader = init(
                valtiny=args.valtiny, no_aug=args.no_aug,
                batch_size=args.batch_size, state=1, split=args.split, input_sizes=input_sizes, data_set=args.dataset,
                sets_id=args.sets_id, mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
            after_loading()
            x = self_train_schedule(writer=writer, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                                    pseudo_labeled_loader=pseudo_labeled_loader, validation_loader=val_loader,
                                    reference_loader=reference_loader, is_mixed_precision=args.mixed_precision,
                                    device=device, criterion=criterion, net=net, optimizer=optimizer,
                                    lr_scheduler=lr_scheduler, st_lr=args.st_lr, init_epochs=args.init_epochs,
                                    st_epochs=args.st_epochs, num_iters=args.iters, optimizer_init=st_optimizer_init,
                                    init_label_ratio=args.init_label_ratio, det_label_ratio=args.det_label_ratio,
                                    input_sizes=input_sizes, categories=categories, num_classes=num_classes,
                                    val_num_steps=args.val_num_steps, exp_name=args.exp_name)

        else:
            # Support unsupervised learning here if that's what you want
            # But we do not think that works, yet...
            raise ValueError

        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint
            os.rename('temp.pt', exp_name + '.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove('temp.pt')

        writer.close()

        with open('log.txt', 'a') as f:
            f.write(exp_name + ': ' + str(x) + '\n')
