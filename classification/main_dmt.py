import os
import time
import torch
import argparse
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.wideresnet import wrn_28_2
from utils.common import num_classes_cifar10, mean_cifar10, std_cifar10, input_sizes_cifar10, base_cifar10, \
    load_checkpoint, save_checkpoint, EMA, rank_label_confidence
from utils.datasets import CIFAR10
from utils.mixup import mixup_data
from utils.losses import SigmoidAscendingMixupDMTLoss as MixupDynamicMutualLoss, DynamicMutualLoss
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
from utils.randomrandaugment import RandomRandAugment
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler


def get_transforms(auto_augment, input_sizes, m, mean, n, std):
    if auto_augment:
        # AutoAugment + Cutout
        train_transforms = Compose([
            RandomCrop(size=input_sizes, padding=4, fill=128),
            RandomHorizontalFlip(p=0.5),
            CIFAR10Policy(),
            ToTensor(),
            Normalize(mean=mean, std=std),
            Cutout(n_holes=1, length=16)
        ])
    else:
        # RandAugment + Cutout
        train_transforms = Compose([
            RandomCrop(size=input_sizes, padding=4, fill=128),
            RandomHorizontalFlip(p=0.5),
            RandomRandAugment(n=n, m_max=m),  # This version includes cutout
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    test_transforms = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return test_transforms, train_transforms


def generate_pseudo_labels(net, device, loader, label_ratio, num_images, filename, is_mixed_precision):
    k = rank_label_confidence(net=net, device=device, loader=loader, ratio=label_ratio, num_images=num_images,
                              is_mixed_precision=is_mixed_precision)
    print(k)
    # 1 forward pass (build pickle file)
    selected_files = None
    selected_predictions = None
    net.eval()
    with torch.no_grad():
        for images, original_file in tqdm(loader):
            # Inference
            images = images.to(device)
            with autocast(is_mixed_precision):
                outputs = net(images)
                temp = torch.nn.functional.softmax(input=outputs, dim=-1)  # ! softmax
                pseudo_probabilities = temp.max(dim=-1).values

            # Select
            temp_predictions = temp[pseudo_probabilities > k].cpu().numpy()
            temp_files = original_file[pseudo_probabilities.cpu() > k].numpy()

            # Append
            selected_files = temp_files if selected_files is None else np.concatenate((selected_files, temp_files))
            selected_predictions = temp_predictions if selected_predictions is None else \
                np.concatenate((selected_predictions, temp_predictions))

    # Save (label format: softmax results in numpy)
    with open(filename, 'wb') as f:
        pickle.dump({'data': selected_files, 'labels': selected_predictions}, f)


def init(mean, std, input_sizes, base, num_workers, prefix, val_set, train, batch_size_labeled, batch_size_pseudo,
         auto_augment=False, n=1, m=1, dataset='cifar10'):
    test_transforms, train_transforms = get_transforms(auto_augment, input_sizes, m, mean, n, std)

    # Data sets
    if dataset == 'cifar10':
        unlabeled_set = CIFAR10(root=base, set_name=prefix + '_unlabeled', transform=test_transforms, label=False)
        if train:
            labeled_set = CIFAR10(root=base, set_name=prefix + '_labeled', transform=train_transforms, label=True)
            pseudo_labeled_set = CIFAR10(root=base, set_name=prefix + '_pseudo', transform=train_transforms,
                                         label=True)
            val_set = CIFAR10(root=base, set_name=val_set, transform=test_transforms, label=True)
    else:
        raise NotImplementedError

    # Data loaders
    unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size_labeled,
                                                   num_workers=num_workers * 4, shuffle=False)
    if train:
        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size_labeled,
                                                 num_workers=num_workers, shuffle=False)
        labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set, batch_size=batch_size_labeled,
                                                     num_workers=num_workers, shuffle=True)
        pseudo_labeled_loader = torch.utils.data.DataLoader(dataset=pseudo_labeled_set, batch_size=batch_size_pseudo,
                                                            num_workers=num_workers * 3, shuffle=True)
    else:
        val_loader = None
        labeled_loader = None
        pseudo_labeled_loader = None

    return labeled_loader, unlabeled_loader, pseudo_labeled_loader, val_loader, unlabeled_set.__len__()


def test(loader, device, net, fine_grain=False, is_mixed_precision=False):
    # Evaluate
    net.eval()
    test_correct = 0
    fine_grain_correct = 0.0
    test_all = 0
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(is_mixed_precision):
                output = net(image)
            test_all += target.shape[0]
            if fine_grain:
                predictions = output.softmax(1)
                temp = predictions.max(1)
                indices = temp.indices
                values = temp.values
                fine_grain_correct += values[indices == target].sum().item()
            test_correct += (target == output.argmax(1)).sum().item()

    test_acc = test_correct / test_all * 100
    print('%d images tested.' % int(test_all))
    print('Test accuracy: %.4f' % test_acc)

    if fine_grain:
        fine_grain_acc = fine_grain_correct / test_all * 100
        print('Fine-grained accuracy: %.4f' % fine_grain_acc)
        return fine_grain_acc

    return test_acc


def train(writer, labeled_loader, pseudo_labeled_loader, val_loader, device, criterion, net, optimizer, lr_scheduler,
          num_epochs, tensorboard_prefix, gamma1, gamma2, labeled_weight, start_at, num_classes, decay=0.999,
          alpha=-1, is_mixed_precision=False, loss_freq=10, val_num_steps=None, best_acc=0, fine_grain=False):
    # Define validation and loss value print frequency
    # Pseudo labeled defines epoch
    min_len = len(pseudo_labeled_loader)
    if min_len > loss_freq:
        loss_num_steps = int(min_len / loss_freq)
    else:  # For extremely small sets
        loss_num_steps = min_len
    if val_num_steps is None:
        val_num_steps = min_len

    if is_mixed_precision:
        scaler = GradScaler()

    net.train()

    # Use EMA to report final performance instead of select best checkpoint with valtiny
    ema = EMA(net=net, decay=decay)

    epoch = 0

    # Training
    running_loss = 0.0
    running_stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'gamma1': 0, 'gamma2': 0}
    iter_labeled = iter(labeled_loader)
    while epoch < num_epochs:
        train_correct = 0
        train_all = 0
        time_now = time.time()
        for i, data in enumerate(pseudo_labeled_loader, 0):
            # Pseudo labeled data
            inputs_pseudo, labels_pseudo = data
            inputs_pseudo, labels_pseudo = inputs_pseudo.to(device), labels_pseudo.to(device)
            
            # Hard labels
            probs_pseudo = labels_pseudo.clone().detach()
            labels_pseudo = labels_pseudo.argmax(-1)  # data type?

            # Labeled data
            inputs_labeled, labels_labeled = next(iter_labeled, (0, 0))
            if type(inputs_labeled) == type(labels_labeled) == int:
                iter_labeled = iter(labeled_loader)
                inputs_labeled, labels_labeled = next(iter_labeled, (0, 0))
            inputs_labeled, labels_labeled = inputs_labeled.to(device), labels_labeled.to(device)

            # To probabilities (in fact, just one-hot)
            probs_labeled = torch.nn.functional.one_hot(labels_labeled.clone().detach(), num_classes=num_classes) \
                .float()

            # Combine
            inputs = torch.cat([inputs_pseudo, inputs_labeled])
            labels = torch.cat([labels_pseudo, labels_labeled])
            probs = torch.cat([probs_pseudo, probs_labeled])
            optimizer.zero_grad()
            train_all += labels.shape[0]

            # mixup data within the batch
            if alpha != -1:
                dynamic_weights, stats = criterion.dynamic_weights_calc(
                    net=net, inputs=inputs, targets=probs,
                    split_index=inputs_pseudo.shape[0], labeled_weight=labeled_weight)
                inputs, dynamic_weights, labels_a, labels_b, lam = mixup_data(x=inputs, w=dynamic_weights, y=labels,
                                                                              alpha=alpha, keep_max=True)
            with autocast(is_mixed_precision):
                outputs = net(inputs)

            if alpha != -1:
                # Pseudo training accuracy & interesting loss
                predicted = outputs.argmax(1)
                train_correct += (lam * (predicted == labels_a).sum().float().item()
                                  + (1 - lam) * (predicted == labels_b).sum().float().item())
                loss, true_loss = criterion(pred=outputs, y_a=labels_a, y_b=labels_b, lam=lam,
                                            dynamic_weights=dynamic_weights)
            else:
                train_correct += (labels == outputs.argmax(1)).sum().item()
                loss, true_loss, stats = criterion(inputs=outputs, targets=probs, split_index=inputs_pseudo.shape[0],
                                                   gamma1=gamma1, gamma2=gamma2)

            if is_mixed_precision:
                accelerator.backward(scaler.scale(loss))
                scaler.step(optimizer)
                scaler.update()
            else:
                accelerator.backward(loss)
                optimizer.step()
            criterion.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # EMA update
            ema.update(net=net)

            # Logging
            running_loss += true_loss
            for key in stats.keys():
                running_stats[key] += stats[key]
            current_step_num = int(epoch * len(pseudo_labeled_loader) + i + 1)
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / loss_num_steps))
                writer.add_scalar(tensorboard_prefix + 'training loss',
                                  running_loss / loss_num_steps,
                                  current_step_num)
                running_loss = 0.0
                for key in stats.keys():
                    print('[%d, %d] ' % (epoch + 1, i + 1) + key + ' : %.4f' % (running_stats[key] / loss_num_steps))
                    writer.add_scalar(tensorboard_prefix + key,
                                      running_stats[key] / loss_num_steps,
                                      current_step_num)
                    running_stats[key] = 0.0

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1) or \
               current_step_num == num_epochs * len(pseudo_labeled_loader) - 1:
                # Apex bug https://github.com/NVIDIA/apex/issues/706, fixed in PyTorch1.6, kept here for BC
                test_acc = test(loader=val_loader, device=device, net=net, fine_grain=fine_grain,
                                is_mixed_precision=is_mixed_precision)
                writer.add_scalar(tensorboard_prefix + 'test accuracy',
                                  test_acc,
                                  current_step_num)
                net.train()

                # Record best model(Straight to disk)
                if test_acc >= best_acc:
                    best_acc = test_acc
                    save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    is_mixed_precision=is_mixed_precision)

        # Evaluate training accuracies (same metric as validation, but must be on-the-fly to save time)
        train_acc = train_correct / train_all * 100
        print('Train accuracy: %.4f' % train_acc)

        writer.add_scalar(tensorboard_prefix + 'train accuracy',
                          train_acc,
                          epoch + 1)

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))

    ema.fill_in_bn(state_dict=net.state_dict())
    save_checkpoint(net=ema, optimizer=None, lr_scheduler=None, is_mixed_precision=False,
                    filename='temp-ema.pt')
    return best_acc


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0 && torchvision 0.7.0')
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Train/Evaluate on Cifar10 (default: cifar10)')
    parser.add_argument('--val-num-steps', type=int, default=1000,
                        help='How many steps between validations (default: 1000)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--gamma1', type=float, default=1,
                        help='Gamma for entropy minimization in agreement (default: 1)')
    parser.add_argument('--gamma2', type=float, default=1,
                        help='Gamma for learning in disagreement (default: 1)')
    parser.add_argument('--aa', action='store_true', default=False,
                        help='Use AutoAugment instead of RandAugment (default: False)')
    parser.add_argument('--n', type=int, default=1,
                        help='N in RandAugment (default: 1)')
    parser.add_argument('--m', type=int, default=10,
                        help='Max M in RandAugment (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Alpha for mixup, -1 -> no mixup (default: 0.75)')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate (default: 0.2)')
    parser.add_argument('--labeled-weight', type=float, default=1,
                        help='Weight for labeled loss (default: 1)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--start-at', type=int, default=0,
                        help='State dynamic weighting at what epoch (default: 150)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for loading (default: 1)')
    parser.add_argument('--batch-size-labeled', type=int, default=64,
                        help='Batch size for labeled data (default: 64)')
    parser.add_argument('--batch-size-pseudo', type=int, default=448,
                        help='Batch size for pseudo labeled data (default: 448)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='Save model (default: True)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--valtiny', action='store_true', default=False,
                        help='Use valtiny as validation/Directly use the test set (default: False)')
    parser.add_argument('--fine-grain', action='store_true', default=False,
                        help='Use fine-grain testing, i.e. 90% correct counts as 0.9 (default: False)')
    parser.add_argument('--labeling', action='store_true', default=False,
                        help='Just pseudo labeling (default: False)')
    parser.add_argument('--label-ratio', type=float, default=1,
                        help='Pseudo labeling ratio (default: 1)')
    parser.add_argument('--train-set', type=str, default='400_seed1',
                        help='The training set file name prefix')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Training begins from a previous checkpoint')
    args = parser.parse_args()
    with open(args.exp_name + '_cfg.txt', 'w') as f:
        f.write(str(vars(args)))

    # Basic configurations
    exp_name = str(int(time.time()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True  # Might hurt performance
    # torch.backends.cudnn.benchmark = False  # Might hurt performance
    if args.exp_name != 'auto':
        exp_name = args.exp_name
    # device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    accelerator = Accelerator(split_batches=True)
    device = accelerator.device
    if args.valtiny:
        val_set = 'valtiny_seed1'
    else:
        val_set = 'test'
    if args.dataset == 'cifar10':
        num_classes = num_classes_cifar10
        mean = mean_cifar10
        std = std_cifar10
        input_sizes = input_sizes_cifar10
        base = base_cifar10
    else:
        raise ValueError

    net = wrn_28_2(num_classes=num_classes)
    print(device)
    net.to(device)

    params_to_optimize = net.parameters()
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)

    labeled_loader, unlabeled_loader, pseudo_labeled_loader, val_loader, num_images = init(
        batch_size_labeled=args.batch_size_labeled, mean=mean, base=base, prefix=args.train_set, val_set=val_set,
        dataset=args.dataset, n=args.n, m=args.m, auto_augment=args.aa, input_sizes=input_sizes, std=std,
        num_workers=args.num_workers, batch_size_pseudo=args.batch_size_pseudo, train=False if args.labeling else True)

    net, optimizer, labeled_loader, pseudo_labeled_loader = accelerator.prepare(net, optimizer,
                                                                                labeled_loader,
                                                                                pseudo_labeled_loader)

    # Pseudo labeling
    if args.labeling:
        time_now = time.time()
        sub_base = CIFAR10.base_folder
        filename = os.path.join(base, sub_base, args.train_set + '_pseudo')
        generate_pseudo_labels(net=net, device=device, loader=unlabeled_loader, filename=filename,
                               label_ratio=args.label_ratio, num_images=num_images,
                               is_mixed_precision=args.mixed_precision)
        print('Pseudo labeling time: %.2fs' % (time.time() - time_now))
    else:
        # Mutual-training
        if args.alpha == -1:
            criterion = DynamicMutualLoss()
        else:
            criterion = MixupDynamicMutualLoss(gamma1=args.gamma1, gamma2=args.gamma2,
                                               T_max=args.epochs * len(pseudo_labeled_loader))
        writer = SummaryWriter('logs/' + exp_name)

        best_acc = test(loader=val_loader, device=device, net=net, fine_grain=args.fine_grain,
                        is_mixed_precision=args.mixed_precision)
        save_checkpoint(net=net, optimizer=None, lr_scheduler=None, is_mixed_precision=args.mixed_precision)
        print('Original acc: ' + str(best_acc))

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=args.epochs * len(pseudo_labeled_loader))
        # lr_scheduler = None

        # Retraining (i.e. fine-tuning)
        best_acc = train(writer=writer, labeled_loader=labeled_loader, pseudo_labeled_loader=pseudo_labeled_loader,
                         val_loader=val_loader, device=device, criterion=criterion, net=net, optimizer=optimizer,
                         lr_scheduler=lr_scheduler, fine_grain=args.fine_grain, alpha=args.alpha,
                         num_epochs=args.epochs, is_mixed_precision=args.mixed_precision, best_acc=best_acc,
                         val_num_steps=args.val_num_steps, tensorboard_prefix='', start_at=args.start_at,
                         labeled_weight=args.labeled_weight, gamma1=args.gamma1, gamma2=args.gamma2,
                         num_classes=num_classes)

        # Tidy up
        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint
            os.rename('temp.pt', exp_name + '.pt')
            os.rename('temp-ema.pt', exp_name + '--ema.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove('temp.pt')
            os.remove('temp-ema.pt')
        writer.close()

        with open('log.txt', 'a') as f:
            f.write(exp_name + ': ' + str(best_acc) + '\n')
