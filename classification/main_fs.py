import os
import time
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
from utils.randomrandaugment import RandomRandAugment
from models.wideresnet import wrn_28_2
from utils.common import num_classes_cifar10, mean_cifar10, std_cifar10, input_sizes_cifar10, base_cifar10, \
                         load_checkpoint, save_checkpoint, EMA
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy
from utils.datasets import CIFAR10
from utils.mixup import mixup_criterion, mixup_data
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler


def init(batch_size, state, mean, std, input_sizes, base, num_workers, train_set, val_set, rand_augment=True,
         n=1, m=1, dataset='cifar10'):
    # # Original transforms
    # train_transforms = Compose([
    #     Pad(padding=4, padding_mode='reflect'),
    #     RandomHorizontalFlip(p=0.5),
    #     RandomCrop(size=input_sizes),
    #     ToTensor(),
    #     Normalize(mean=mean, std=std)
    # ])

    if rand_augment:
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
    else:
        # AutoAugment + Cutout
        train_transforms = Compose([
            RandomCrop(size=input_sizes, padding=4, fill=128),
            RandomHorizontalFlip(p=0.5),
            CIFAR10Policy(),
            ToTensor(),
            Normalize(mean=mean, std=std),
            Cutout(n_holes=1, length=16)
        ])
        test_transforms = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    # Data sets
    if dataset == 'cifar10':
        if state == 1:
            train_set = CIFAR10(root=base, set_name=train_set, transform=train_transforms, label=True)
        test_set = CIFAR10(root=base, set_name=val_set, transform=test_transforms, label=True)
    else:
        raise NotImplementedError

    # Data loaders
    if state == 1:
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                   num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,
                                              num_workers=num_workers * 2, shuffle=False)
    if state == 1:
        return train_loader, test_loader
    else:
        return test_loader


def train(writer, train_loader, val_loader, device, criterion, net, optimizer, lr_scheduler, num_epochs, log_file,
          alpha=None, is_mixed_precision=False, loss_freq=10, val_num_steps=None, best_acc=0, fine_grain=False,
          decay=0.999):
    # Define validation and loss value print frequency
    if len(train_loader) > loss_freq:
        loss_num_steps = int(len(train_loader) / loss_freq)
    else:  # For extremely small sets
        loss_num_steps = len(train_loader)
    if val_num_steps is None:
        val_num_steps = len(train_loader)

    if is_mixed_precision:
        scaler = GradScaler()

    net.train()

    # Use EMA to report final performance instead of select best checkpoint with valtiny
    ema = EMA(net=net, decay=decay)

    epoch = 0

    # Training
    running_loss = 0.0
    while epoch < num_epochs:
        train_correct = 0
        train_all = 0
        time_now = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            train_all += labels.shape[0]

            # mixup data within the batch
            if alpha is not None:
                inputs, labels_a, labels_b, lam = mixup_data(x=inputs, y=labels, alpha=alpha)

            with autocast(is_mixed_precision):
                outputs = net(inputs)

            if alpha is not None:
                # Pseudo training accuracy & interesting loss
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                predicted = outputs.argmax(1)
                train_correct += (lam * (predicted == labels_a).sum().float().item()
                                  + (1 - lam) * (predicted == labels_b).sum().float().item())
            else:
                train_correct += (labels == outputs.argmax(1)).sum().item()
                loss = criterion(outputs, labels)

            if is_mixed_precision:
                accelerator.backward(scaler.scale(loss))
                scaler.step(optimizer)
                scaler.update()
            else:
                accelerator.backward(loss)
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # EMA update
            ema.update(net=net)

            # Logging
            running_loss += loss.item()
            current_step_num = int(epoch * len(train_loader) + i + 1)
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / loss_num_steps))
                writer.add_scalar('training loss',
                                  running_loss / loss_num_steps,
                                  current_step_num)
                running_loss = 0.0

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1) or \
               current_step_num == num_epochs * len(train_loader) - 1:
                # Apex bug https://github.com/NVIDIA/apex/issues/706, fixed in PyTorch1.6, kept here for BC
                test_acc = test(loader=val_loader, device=device, net=net, fine_grain=fine_grain,
                                is_mixed_precision=is_mixed_precision)
                writer.add_scalar('test accuracy',
                                  test_acc,
                                  current_step_num)
                net.train()

                # Record best model(Straight to disk)
                if test_acc > best_acc:
                    best_acc = test_acc
                    save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    is_mixed_precision=is_mixed_precision, filename=log_file + '_temp.pt')

        # Evaluate training accuracies (same metric as validation, but must be on-the-fly to save time)
        train_acc = train_correct / train_all * 100
        print('Train accuracy: %.4f' % train_acc)

        writer.add_scalar('train accuracy',
                          train_acc,
                          epoch + 1)

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))

    ema.fill_in_bn(state_dict=net.state_dict())
    save_checkpoint(net=ema, optimizer=None, lr_scheduler=None, is_mixed_precision=False,
                    filename=log_file + '_temp-ema.pt')
    return best_acc


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
    parser.add_argument('--ra', action='store_true', default=False,
                        help='Use RandAugment instead of AutoAugment (default: False)')
    parser.add_argument('--n', type=int, default=3,
                        help='N in RandAugment (default: 3)')
    parser.add_argument('--m', type=int, default=7,
                        help='M in RandAugment (default: 7)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Alpha for mixup, None -> no mixup (default: None)')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for loading (default: 2)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size (default: 128)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='Save model (default: True)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--valtiny', action='store_true', default=False,
                        help='Use valtiny as validation/Directly use the test set (default: False)')
    parser.add_argument('--fine-grain', action='store_true', default=False,
                        help='Use fine-grain testing, i.e. 90% correct counts as 0.9 (default: False)')
    parser.add_argument('--train-set', type=str, default='trainval_seed1',
                        help='The training set file name')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Training begins from a previous checkpoint/Test on this')
    parser.add_argument('--log', type=str, default='log',
                        help='Log file name')
    parser.add_argument('--state', type=int, default=1,
                        help="Final test(2)/Fully-supervised training(1)")
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

    # Define optimizer
    params_to_optimize = net.parameters()
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    # Testing
    if args.state == 2:
        net, optimizer = accelerator.prepare(net, optimizer)
        test_loader = init(batch_size=args.batch_size, state=2, mean=mean, std=std, train_set=None, val_set=val_set,
                           input_sizes=input_sizes, base=base, num_workers=args.num_workers, dataset=args.dataset)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        x = test(loader=test_loader, device=device, net=net, fine_grain=args.fine_grain,
                 is_mixed_precision=args.mixed_precision)
        with open(args.log + '.txt', 'a') as f:
            f.write('test: ' + str(x) + '\n')

    # Training
    elif args.state == 1:
        x = 0
        criterion = torch.nn.CrossEntropyLoss()
        writer = SummaryWriter('logs/' + exp_name)
        train_loader, val_loader = init(batch_size=args.batch_size, state=1, mean=mean, std=std, base=base,
                                        train_set=args.train_set, val_set=val_set, n=args.n, m=args.m,
                                        rand_augment=args.ra,
                                        input_sizes=input_sizes, num_workers=args.num_workers, dataset=args.dataset)
        net, optimizer, train_loader = accelerator.prepare(net, optimizer, train_loader)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=args.epochs * len(train_loader))
        # lr_scheduler = None
        if args.continue_from is not None:
            load_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                            is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        x = train(writer=writer, train_loader=train_loader, val_loader=val_loader, device=device, criterion=criterion,
                  net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, fine_grain=args.fine_grain, alpha=args.alpha,
                  num_epochs=args.epochs, is_mixed_precision=args.mixed_precision, val_num_steps=args.val_num_steps,
                  log_file=args.log)

        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint
            os.rename(args.log + '_temp.pt', exp_name + '.pt')
            os.rename(args.log + '_temp-ema.pt', exp_name + '--ema.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove(args.log + '_temp.pt')
            os.remove(args.log + '_temp-ema.pt')

        writer.close()

        with open(args.log + '.txt', 'a') as f:
            f.write(exp_name + ': ' + str(x) + '\n')

    else:
        raise ValueError
