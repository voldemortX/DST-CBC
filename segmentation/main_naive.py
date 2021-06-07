# Recommend training a supervised baseline first,
# then conduct self-training from it to avoid mini-batch size issues

import os
import time
import copy
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from models.segmentation.segmentation import deeplab_v2
from utils.datasets import StandardSegmentationDataset
from utils.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, RandomResize, Resize, LabelMap, \
    ZeroPad, Compose, RandomScale
from torch.utils.tensorboard import SummaryWriter
from utils.common import colors_city, colors_voc, categories_voc, categories_city, sizes_city, sizes_voc, \
    num_classes_voc, num_classes_city, coco_mean, coco_std, imagenet_mean, imagenet_std, ConfusionMatrix, \
    load_checkpoint, save_checkpoint, generate_class_balanced_pseudo_labels, base_city, base_voc, label_id_map_city
from utils.losses import DynamicNaiveLoss as DynamicMutualLoss
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler


def init(batch_size_labeled, batch_size_pseudo, state, split, input_sizes, sets_id, std, mean,
         keep_scale, reverse_channels, data_set, valtiny, no_aug):
    # Return data_loaders/data_loader
    # depending on whether the state is
    # 0: Pseudo labeling
    # 1: Semi-supervised training
    # 2: Fully-supervised training
    # 3: Just testing

    # For labeled set divisions
    split_u = split.replace('-r', '')
    split_u = split_u.replace('-l', '')

    # Transformations (compatible with unlabeled data/pseudo labeled data)
    # ! Can't use torchvision.Transforms.Compose
    if data_set == 'voc':
        base = base_voc
        workers = 4
        transform_train = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
             RandomScale(min_scale=0.5, max_scale=1.5),
             RandomCrop(size=input_sizes[0]),
             RandomHorizontalFlip(flip_prob=0.5),
             Normalize(mean=mean, std=std)])
        if no_aug:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
                 Normalize(mean=mean, std=std)])
        else:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 RandomScale(min_scale=0.5, max_scale=1.5),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
        # transform_pseudo = Compose(
        #     [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
        #      Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
        #      Normalize(mean=mean, std=std)])
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             ZeroPad(size=input_sizes[2]),
             Normalize(mean=mean, std=std)])
    elif data_set == 'city':  # All the same size (whole set is down-sampled by 2)
        base = base_city
        workers = 8
        transform_train = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
             Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
             RandomScale(min_scale=0.5, max_scale=1.5),
             RandomCrop(size=input_sizes[0]),
             RandomHorizontalFlip(flip_prob=0.5),
             Normalize(mean=mean, std=std),
             LabelMap(label_id_map_city)])
        if no_aug:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
                 Normalize(mean=mean, std=std)])
        else:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
                 RandomScale(min_scale=0.5, max_scale=1.5),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
        # transform_pseudo = Compose(
        #     [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
        #      Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
        #      Normalize(mean=mean, std=std),
        #      LabelMap(label_id_map_city)])
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
             Normalize(mean=mean, std=std),
             LabelMap(label_id_map_city)])
    else:
        base = ''

    # Not the actual test set (i.e.validation set)
    test_set = StandardSegmentationDataset(root=base, image_set='valtiny' if valtiny else 'val',
                                           transforms=transform_test, label_state=0, data_set=data_set)
    val_loader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=batch_size_labeled + batch_size_pseudo,
                                             num_workers=workers, shuffle=False)

    # Testing
    if state == 3:
        return val_loader
    else:
        # Fully-supervised training
        if state == 2:
            labeled_set = StandardSegmentationDataset(root=base, image_set=(str(split) + '_labeled_' + str(sets_id)),
                                                      transforms=transform_train, label_state=0, data_set=data_set)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set, batch_size=batch_size_labeled,
                                                         num_workers=workers, shuffle=True)
            return labeled_loader, val_loader

        # Semi-supervised training
        elif state == 1:
            pseudo_labeled_set = StandardSegmentationDataset(root=base, data_set=data_set, mask_type='.npy',
                                                             image_set=(str(split_u) + '_unlabeled_' + str(sets_id)),
                                                             transforms=transform_train_pseudo, label_state=1)
            labeled_set = StandardSegmentationDataset(root=base, data_set=data_set,
                                                      image_set=(str(split) + '_labeled_' + str(sets_id)),
                                                      transforms=transform_train, label_state=0)
            pseudo_labeled_loader = torch.utils.data.DataLoader(dataset=pseudo_labeled_set,
                                                                batch_size=batch_size_pseudo,
                                                                num_workers=workers, shuffle=True)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set,
                                                         batch_size=batch_size_labeled,
                                                         num_workers=workers, shuffle=True)
            return labeled_loader, pseudo_labeled_loader, val_loader

        else:
            # Labeling
            unlabeled_set = StandardSegmentationDataset(root=base, data_set=data_set, mask_type='.npy',
                                                        image_set=(str(split_u) + '_unlabeled_' + str(sets_id)),
                                                        transforms=transform_test, label_state=2)
            unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size_labeled,
                                                           num_workers=workers,
                                                           shuffle=False)
            return unlabeled_loader


def train(writer, loader_c, loader_sup, validation_loader, device, criterion, net, optimizer, lr_scheduler,
          num_epochs, is_mixed_precision, with_sup, num_classes, categories, input_sizes,
          val_num_steps=1000, loss_freq=10, tensorboard_prefix='', best_mIoU=0):
    #######
    # c for carry (pseudo labeled), sup for support (labeled with ground truth) -_-
    # Don't ask me why
    #######
    # Poly training schedule
    # Epoch length measured by "carry" (c) loader
    # Batch ratio is determined by loaders' own batch size
    # Validate and find the best snapshot per val_num_steps
    loss_num_steps = int(len(loader_c) / loss_freq)
    net.train()
    epoch = 0
    if with_sup:
        iter_sup = iter(loader_sup)

    if is_mixed_precision:
        scaler = GradScaler()

    # Training
    running_stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': 0.0}
    while epoch < num_epochs:
        conf_mat = ConfusionMatrix(num_classes)
        time_now = time.time()
        for i, data in enumerate(loader_c, 0):
            # Combine loaders (maybe just alternate training will work)
            if with_sup:
                inputs_c, labels_c = data
                inputs_sup, labels_sup = next(iter_sup, (0, 0))
                if type(inputs_sup) == type(labels_sup) == int:
                    iter_sup = iter(loader_sup)
                    inputs_sup, labels_sup = next(iter_sup, (0, 0))

                # Formatting (prob: label + max confidence, label: just label)
                float_labels_sup = labels_sup.clone().float().unsqueeze(1)
                probs_sup = torch.cat([float_labels_sup, torch.ones_like(float_labels_sup)], dim=1)
                probs_c = labels_c.clone()
                labels_c = labels_c[:, 0, :, :].long()

                # Concatenating
                inputs = torch.cat([inputs_c, inputs_sup])
                labels = torch.cat([labels_c, labels_sup])
                probs = torch.cat([probs_c, probs_sup])

                probs = probs.to(device)
            else:
                inputs, labels = data

            # Normal training
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast(is_mixed_precision):
                outputs = net(inputs)['out']
                outputs = torch.nn.functional.interpolate(outputs, size=input_sizes[0], mode='bilinear', align_corners=True)
                conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())

                if with_sup:
                    loss, stats = criterion(outputs, probs, inputs_c.shape[0])
                else:
                    loss, stats = criterion(outputs, labels)

            if is_mixed_precision:
                accelerator.backward(scaler.scale(loss))
                scaler.step(optimizer)
                scaler.update()
            else:
                accelerator.backward(loss)
                optimizer.step()
            lr_scheduler.step()

            # Logging
            for key in stats.keys():
                running_stats[key] += stats[key]
            current_step_num = int(epoch * len(loader_c) + i + 1)
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                for key in running_stats.keys():
                    print('[%d, %d] ' % (epoch + 1, i + 1) + key + ' : %.4f' % (running_stats[key] / loss_num_steps))
                    writer.add_scalar(tensorboard_prefix + key,
                                      running_stats[key] / loss_num_steps,
                                      current_step_num)
                    running_stats[key] = 0.0

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1) or \
                current_step_num == num_epochs * len(loader_c) - 1:
                # Apex bug https://github.com/NVIDIA/apex/issues/706, fixed in PyTorch1.6, kept here for BC
                test_pixel_accuracy, test_mIoU = test_one_set(loader=validation_loader, device=device, net=net,
                                                              num_classes=num_classes, categories=categories,
                                                              output_size=input_sizes[2],
                                                              is_mixed_precision=is_mixed_precision)
                writer.add_scalar(tensorboard_prefix + 'test pixel accuracy',
                                  test_pixel_accuracy,
                                  current_step_num)
                writer.add_scalar(tensorboard_prefix + 'test mIoU',
                                  test_mIoU,
                                  current_step_num)
                net.train()

                # Record best model(Straight to disk)
                if test_mIoU > best_mIoU:
                    best_mIoU = test_mIoU
                    save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    is_mixed_precision=is_mixed_precision)

        # Evaluate training accuracies(same metric as validation, but must be on-the-fly to save time)
        acc_global, acc, iu = conf_mat.compute()
        print(categories)
        print((
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100))

        train_pixel_acc = acc_global.item() * 100
        train_mIoU = iu.mean().item() * 100
        writer.add_scalar(tensorboard_prefix + 'train pixel accuracy',
                          train_pixel_acc,
                          epoch + 1)
        writer.add_scalar(tensorboard_prefix + 'train mIoU',
                          train_mIoU,
                          epoch + 1)

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))

    return best_mIoU


# Copied and modified from torch/vision/references/segmentation
def test_one_set(loader, device, net, categories, num_classes, output_size, is_mixed_precision):
    # Evaluate on 1 data_loader (cudnn impact < 0.003%)
    net.eval()
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(is_mixed_precision):
                output = net(image)['out']
                output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
                conf_mat.update(target.flatten(), output.argmax(1).flatten())

    acc_global, acc, iu = conf_mat.compute()
    print(categories)
    print((
        'global correct: {:.2f}\n'
        'average row correct: {}\n'
        'IoU: {}\n'
        'mean IoU: {:.2f}').format(
        acc_global.item() * 100,
        ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100))

    return acc_global.item() * 100, iu.mean().item() * 100


def after_loading():
    global lr_scheduler

    # The "poly" policy, variable names are confusing(May need reimplementation)
    if not args.labeling:
        if args.state == 2:
            len_loader = (len(labeled_loader) * args.epochs)
        else:
            len_loader = (len(pseudo_labeled_loader) * args.epochs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / len_loader) ** 0.9)

    # Resume training?
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0 && torchvision 0.7.0')
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city) (default: voc)')
    parser.add_argument('--gamma1', type=float, default=0,
                        help='Gamma for entropy minimization in agreement (default: 0)')
    parser.add_argument('--gamma2', type=float, default=0,
                        help='Gamma for learning in disagreement (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--val-num-steps', type=int, default=500,
                        help='How many steps between validations (default: 500)')
    parser.add_argument('--label-ratio', type=float, default=0.2,
                        help='Initial labeling ratio (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.002)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for the fully-supervised initialization (default: 30)')
    parser.add_argument('--batch-size-labeled', type=int, default=1,
                        help='Batch size for labeled data (default: 1)')
    parser.add_argument('--batch-size-pseudo', type=int, default=7,
                        help='Batch size for pseudo labeled data (default: 7)')
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
    parser.add_argument('--labeling', action='store_true', default=False,
                        help='Just pseudo labeling (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Self-training begins from a previous checkpoint/Test on this')
    parser.add_argument('--train-set', type=str, default='1',
                        help='e.g. 1:7(8), 1:3(4), 1:1(2), 1:0(1) labeled/unlabeled split (default: 1)')
    parser.add_argument('--sets-id', type=int, default=0,
                        help='Different random splits(0/1/2) (default: 0)')
    parser.add_argument('--state', type=int, default=1,
                        help="Final test(3)/Fully-supervised training(2)/Semi-supervised training(1)")
    args = parser.parse_args()

    # Basic configurations
    exp_name = str(int(time.time()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True  # Might hurt performance
    # torch.backends.cudnn.benchmark = False  # Might hurt performance
    if args.exp_name != 'auto':
        exp_name = args.exp_name
    with open(exp_name + '_cfg.txt', 'w') as f:
        f.write(str(vars(args)))
    # device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    accelerator = Accelerator(split_batches=True)
    device = accelerator.device
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

    # Just to be safe (a little bit more memory, by all means, save it to disk if you want)
    if args.state == 1:
        st_optimizer_init = copy.deepcopy(optimizer.state_dict())

    # Testing
    if args.state == 3:
        net, optimizer = accelerator.prepare(net, optimizer)
        test_loader = init(batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                           state=3, split=None, valtiny=args.valtiny, no_aug=args.no_aug,
                           input_sizes=input_sizes, data_set=args.dataset, sets_id=args.sets_id,
                           mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        test_one_set(loader=test_loader, device=device, net=net, categories=categories, num_classes=num_classes,
                     output_size=input_sizes[2], is_mixed_precision=args.mixed_precision)
    else:
        x = 0
        criterion = DynamicMutualLoss(ignore_index=255)
        writer = SummaryWriter('logs/' + exp_name)

        # Only fully-supervised training
        if args.state == 2:
            labeled_loader, val_loader = init(batch_size_labeled=args.batch_size_labeled,
                                              batch_size_pseudo=args.batch_size_pseudo, sets_id=args.sets_id,
                                              valtiny=args.valtiny,
                                              state=2, split=args.train_set, input_sizes=input_sizes,
                                              data_set=args.dataset,
                                              mean=mean, std=std, keep_scale=keep_scale, no_aug=args.no_aug,
                                              reverse_channels=reverse_channels)
            after_loading()
            net, optimizer, labeled_loader = accelerator.prepare(net, optimizer, labeled_loader)
            x = train(writer=writer, loader_c=labeled_loader, loader_sup=None, validation_loader=val_loader,
                      device=device, criterion=criterion, net=net, optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      num_epochs=args.epochs, categories=categories, num_classes=num_classes,
                      is_mixed_precision=args.mixed_precision, with_sup=False,
                      val_num_steps=args.val_num_steps, input_sizes=input_sizes)

        # Self-training
        elif args.state == 1:
            if args.labeling:
                unlabeled_loader = init(
                    valtiny=args.valtiny, no_aug=args.no_aug, data_set=args.dataset,
                    batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                    state=0, split=args.train_set, input_sizes=input_sizes,
                    sets_id=args.sets_id, mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
                after_loading()
                net, optimizer = accelerator.prepare(net, optimizer)
                time_now = time.time()
                ratio = generate_class_balanced_pseudo_labels(net=net, device=device, loader=unlabeled_loader,
                                                              input_size=input_sizes[2],
                                                              label_ratio=args.label_ratio, num_classes=num_classes)
                print(ratio)
                print('Pseudo labeling time: %.2fs' % (time.time() - time_now))
            else:
                labeled_loader, pseudo_labeled_loader, val_loader = init(
                    valtiny=args.valtiny, no_aug=args.no_aug, data_set=args.dataset,
                    batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                    state=1, split=args.train_set, input_sizes=input_sizes,
                    sets_id=args.sets_id, mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
                after_loading()
                net, optimizer, labeled_loader, pseudo_labeled_loader = accelerator.prepare(net, optimizer,
                                                                                            labeled_loader,
                                                                                            pseudo_labeled_loader)
                x = train(writer=writer, loader_c=pseudo_labeled_loader, loader_sup=labeled_loader,
                          validation_loader=val_loader, lr_scheduler=lr_scheduler,
                          device=device, criterion=criterion, net=net, optimizer=optimizer,
                          num_epochs=args.epochs, categories=categories, num_classes=num_classes,
                          is_mixed_precision=args.mixed_precision, with_sup=True,
                          val_num_steps=args.val_num_steps, input_sizes=input_sizes)

        else:
            # Support unsupervised learning here if that's what you want
            # But we do not think that works, yet...
            raise ValueError

        if not args.labeling:
            # --do-not-save => args.do_not_save = False
            if args.do_not_save:  # Rename the checkpoint
                os.rename('temp.pt', exp_name + '.pt')
            else:  # Since the checkpoint is already saved, it should be deleted
                os.remove('temp.pt')

            writer.close()

            with open('log.txt', 'a') as f:
                f.write(exp_name + ': ' + str(x) + '\n')
