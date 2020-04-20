import shutil
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from apex import amp
from PIL import Image
from torchvision_models.segmentation.segmentation import deeplabv2_resnet101
from data_processing import StandardSegmentationDataset, SegmentationLabelsDataset, \
                            base_city, base_voc, label_id_map_city
from transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, RandomResize, Resize, LabelMap, ZeroPad, \
                       Compose


def deeplab_v2(num_classes):
    # Define deeplabV2 with ResNet101(With only ImageNet pretraining)
    return deeplabv2_resnet101(pretrained=False, num_classes=num_classes, recon_loss=False)


# Copied and simplified from torch/vision/references/segmentation to compute mean IoU
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # For pseudo labels(which has 255), just don't let your network predict 255
            k = (a >= 0) & (a < n) & (b != 255)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


# Draw images/labels from tensors
def show(images, is_label, colors, std, mean):
    np_images = images.numpy()
    if is_label:
        # Map to RGB((N, d1, d2) = {0~20, 255} => (N, d1, d2, 3) = {0.0~1.0})
        # As for how I managed this, I literally have no clue,
        # but it seems to be working
        np_images = np_images.reshape((np_images.shape[0], np_images.shape[1], np_images.shape[2], 1))
        np_images = np.tile(np_images, (1, 1, 1, 3))
        np_images[np_images == 255] = len(colors) - 1  # Ignore 255
        np_images = np.array(colors)[np_images[:, :, :, 0]]
        np_images = np_images / 255.0
    else:
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


def init(batch_size, state, split, input_sizes, sets_id, std, mean, keep_scale, reverse_channels, data_set,
         valtiny, no_aug):
    # Return data_loaders/data_loader
    # depending on whether the split is
    # 1: semi-supervised training
    # 2: fully-supervised training
    # 3: Just testing

    # Transformations (compatible with unlabeled data/pseudo labeled data)
    # ! Can't use torchvision.Transforms.Compose
    if data_set == 'voc':
        base = base_voc
        workers = 4
        transform_train = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
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
                 RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
        transform_pseudo = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             Normalize(mean=mean, std=std)])
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             ZeroPad(size=input_sizes[2]),
             Normalize(mean=mean, std=std)])
    elif data_set == 'city':  # All the same size (whole set is down-sampled by 2)
        base = base_city
        workers = 8
        transform_train = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
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
                 RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
        transform_pseudo = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             Normalize(mean=mean, std=std),
             LabelMap(label_id_map_city)])
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
    val_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=workers, shuffle=False)

    # Testing
    if state == 3:
        return val_loader
    else:
        # Fully-supervised training
        if state == 2:
            labeled_set = StandardSegmentationDataset(root=base, image_set=(str(split) + '_labeled_' + str(sets_id)),
                                                      transforms=transform_train, label_state=0, data_set=data_set)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set, batch_size=batch_size,
                                                         num_workers=workers, shuffle=True)
            return labeled_loader, val_loader

        # Semi-supervised training
        elif state == 1:
            pseudo_labeled_set = StandardSegmentationDataset(root=base, data_set=data_set,
                                                             image_set=(str(split) + '_unlabeled_' + str(sets_id)),
                                                             transforms=transform_train_pseudo, label_state=1)
            reference_set = SegmentationLabelsDataset(root=base, image_set=(str(split) + '_unlabeled_' + str(sets_id)),
                                                      data_set=data_set)
            reference_loader = torch.utils.data.DataLoader(dataset=reference_set, batch_size=batch_size,
                                                           num_workers=workers, shuffle=False)
            unlabeled_set = StandardSegmentationDataset(root=base, data_set=data_set,
                                                        image_set=(str(split) + '_unlabeled_' + str(sets_id)),
                                                        transforms=transform_pseudo, label_state=2)
            labeled_set = StandardSegmentationDataset(root=base, data_set=data_set,
                                                      image_set=(str(split) + '_labeled_' + str(sets_id)),
                                                      transforms=transform_train, label_state=0)

            unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size,
                                                           num_workers=workers, shuffle=False)

            pseudo_labeled_loader = torch.utils.data.DataLoader(dataset=pseudo_labeled_set,
                                                                batch_size=int(batch_size / 2),
                                                                num_workers=workers, shuffle=True)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set,
                                                         batch_size=int(batch_size / 2),
                                                         num_workers=workers, shuffle=True)
            return labeled_loader, pseudo_labeled_loader, unlabeled_loader, val_loader, reference_loader

        else:
            # Support unsupervised learning here if that's what you want
            raise ValueError


def train_one_iter(writer, loader_c, loader_sup, validation_loader, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision, with_sup, num_classes, categories, input_sizes, thresholds,
                   val_num_steps=1000, loss_freq=10, tensorboard_prefix='', best_mIoU=0):
    #######
    # c for carry (pseudo labeled), sup for support (labeled with ground truth) -_-
    # Don't ask me why
    #######
    # Poly training schedule
    # Epoch length measured by "carry" (c) loader
    # Batch ratio is determined by loaders' own batch size (loader_c batch:loader_sup batch = 1:1)
    # Validate and find the best snapshot per val_num_steps
    loss_num_steps = int(len(loader_c) / loss_freq)
    net.train()
    epoch = 0
    if with_sup:
        iter_sup = iter(loader_sup)

    # Training
    running_loss = 0.0
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
                # No need to shuffle, right?
                inputs = torch.cat([inputs_c, inputs_sup])
                labels = labels_sup
            else:
                inputs, labels = data

            # Normal training
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)['out']
            outputs = torch.nn.functional.interpolate(outputs, size=input_sizes[0], mode='bilinear', align_corners=True)

            # Online labeling
            if with_sup:
                temp = (outputs[:inputs_c.shape[0]].clone().detach().softmax(dim=1).permute(0, 2, 3, 1) / thresholds)\
                       .max(dim=-1)
                labels_c = temp.indices
                weighted_values_c = temp.values
                labels_c[weighted_values_c < 1] = 255
                labels = torch.cat([labels_c, labels.to(device)])

            conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())

            if with_sup:
                loss, statistics = criterion(outputs, labels, inputs_c.shape[0])
            else:
                loss, statistics = criterion(outputs, labels)

            if is_mixed_precision:
                # 2/3 & 3/3 of mixed precision training with amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Logging
            running_loss += statistics['dl']
            current_step_num = int(epoch * len(loader_c) + i + 1)
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / loss_num_steps))
                writer.add_scalar(tensorboard_prefix + 'training loss',
                                  running_loss / loss_num_steps,
                                  current_step_num)
                running_loss = 0.0

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1) or \
               current_step_num == num_epochs * len(loader_c) - 1:
                # A bug in Apex? https://github.com/NVIDIA/apex/issues/706
                test_pixel_accuracy, test_mIoU = test_one_set(loader=validation_loader, device=device, net=net,
                                                              num_classes=num_classes, categories=categories,
                                                              output_size=input_sizes[2])
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
def test_one_set(loader, device, net, categories, num_classes, output_size):
    # Evaluate on 1 data_loader(cudnn impact < 0.003%)
    net.eval()
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
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


def self_train_schedule(writer, labeled_loader, pseudo_labeled_loader, validation_loader,
                        device, criterion, net, optimizer, lr_scheduler,
                        st_lr, init_epochs, st_epochs, num_iters, optimizer_init, is_mixed_precision,
                        threshold, categories, num_classes, input_sizes, exp_name,
                        val_num_steps=1000):
    # Conduct self-training iterations for semi-supervised & unsupervised training
    # Resume training from checkpoints is only supported for checkpoints made after each training round

    # Start with fully supervised baseline
    best_mIoU = train_one_iter(writer=writer, loader_c=labeled_loader, loader_sup=None,
                               validation_loader=validation_loader, net=net,
                               device=device, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
                               num_epochs=init_epochs, tensorboard_prefix='Supervised initial training ',
                               is_mixed_precision=is_mixed_precision, with_sup=False, thresholds=None,
                               categories=categories, num_classes=num_classes, input_sizes=input_sizes)

    # Self-training phase
    for i in range(0, num_iters):
        # Reset optimizer
        optimizer.load_state_dict(optimizer_init)
        for param_group in optimizer.param_groups:
            param_group['lr'] = st_lr

        # Scheduler for SGD takes up negligible memory (just redeclare them)
        st_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                            lambda x: (1 - x / (len(pseudo_labeled_loader) *
                                                                                st_epochs[i])) ** 0.9)

        thresholds = torch.tensor([(threshold + 0.0001) for _ in range(num_classes)]).to(device)

        # Retraining (i.e. fine-tuning)
        if i == 0 and best_mIoU == 0:
            _, best_mIoU = test_one_set(loader=validation_loader, device=device, net=net,
                                        num_classes=num_classes, categories=categories,
                                        output_size=input_sizes[2])
            save_checkpoint(net=net, optimizer=None, lr_scheduler=None, is_mixed_precision=is_mixed_precision)
            print('Original mIoU: ' + str(best_mIoU))

        tensorboard_prefix = 'Self-training iteration ' + str(i) + ' '
        best_mIoU = train_one_iter(writer=writer, device=device, loader_c=pseudo_labeled_loader,
                                   loader_sup=labeled_loader, validation_loader=validation_loader,
                                   criterion=criterion, net=net, optimizer=optimizer, lr_scheduler=st_lr_scheduler,
                                   num_epochs=st_epochs[i], input_sizes=input_sizes,
                                   is_mixed_precision=is_mixed_precision, with_sup=True,
                                   tensorboard_prefix=tensorboard_prefix, best_mIoU=best_mIoU,
                                   num_classes=num_classes, categories=categories, val_num_steps=val_num_steps,
                                   thresholds=thresholds)

        # Select best model for next iteration (could be from earlier iterations)
        print('loading best model...')
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=is_mixed_precision, filename='temp.pt')
        test_pixel_accuracy, test_mIoU = test_one_set(loader=validation_loader, device=device, net=net,
                                                      num_classes=num_classes, categories=categories,
                                                      output_size=input_sizes[2])
        writer.add_scalar('best test pixel accuracy',
                          test_pixel_accuracy,
                          i + 1)
        writer.add_scalar('best test mIoU',
                          test_mIoU,
                          i + 1)

        # Save some checkpoints that are used for pseudo labeling [! You can comment these 2 lines to save space]
        if i < num_iters - 1:
            shutil.copy('temp.pt', exp_name + '___' + str(i + 1) + '.pt')

    return best_mIoU
