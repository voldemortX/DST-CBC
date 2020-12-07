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
                       Compose, RandomScale
from functional import crop


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
    # depending on whether the state is
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
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             ZeroPad(size=input_sizes[2]),
             Normalize(mean=mean, std=std)])
        transform_pseudo = transform_test
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
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
             Normalize(mean=mean, std=std),
             LabelMap(label_id_map_city)])
        transform_pseudo = transform_test
        
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
                   num_epochs, is_mixed_precision, with_sup, num_classes, categories, input_sizes,
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
                labels_c = torch.cat([labels_c, labels_sup])
                labels = labels_c
            else:
                inputs, labels = data

            # Normal training
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)['out']
            outputs = torch.nn.functional.interpolate(outputs, size=input_sizes[0], mode='bilinear', align_corners=True)
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


def generate_pseudo_labels(net, device, loader, num_classes, input_size, cbst_thresholds=None):
    # Generate pseudo labels and save to disk (negligible time compared with training)
    # Not very sure if there are any cache inconsistency issues (technically this should be fine)
    net.eval()

    # 1 forward pass (hard labels)
    if cbst_thresholds is None:  # Default
        cbst_thresholds = torch.tensor([0.99 for _ in range(num_classes)])
    cbst_thresholds = cbst_thresholds.to(device)
    net.eval()
    labeled_counts = 0
    ignored_counts = 0
    with torch.no_grad():
        for images, file_name_lists, heights, widths in tqdm(loader):
            images = images.to(device)
            outputs = net(images)['out']
            outputs = torch.nn.functional.interpolate(outputs, size=input_size, mode='bilinear', align_corners=True)

            # Generate pseudo labels (d1 x d2)
            for i in range(0, len(file_name_lists)):
                prediction = crop(outputs[i], 0, 0, heights[i], widths[i])  # Back to the original size
                prediction = prediction.softmax(dim=0)  # ! softmax
                temp = prediction.max(dim=0)
                pseudo_label = temp.indices
                values = temp.values
                for j in range(num_classes):
                    pseudo_label[((pseudo_label == j) * (values < cbst_thresholds[j]))] = 255

                # Counting & Saving
                labeled_counts += (pseudo_label != 255).sum().item()
                ignored_counts += (pseudo_label == 255).sum().item()
                pseudo_label = pseudo_label.cpu().numpy().astype(np.uint8)
                if '.png' in file_name_lists[i]:
                    Image.fromarray(pseudo_label).save(file_name_lists[i])
                elif '.npy' in file_name_lists[i]:
                    np.save(file_name_lists[i], pseudo_label)

    # Return overall labeled ratio
    return labeled_counts / (labeled_counts + ignored_counts)


# Reimplemented(all converted to tensor ops) based on yzou2/CRST
def generate_class_balanced_pseudo_labels(net, device, loader, label_ratio, num_classes, input_size,
                                          down_sample_rate=16, buffer_size=100):
    # Max memory usage surge ratio has an upper limit of 2x (caused by array concatenation).
    # Keep a fixed GPU buffer size to achieve a good enough speed-memory trade-off,
    # since casting to cpu is very slow.
    # Note that tensor.expand() does not allocate new memory,
    # and that Python's list consumes at least 3 times the memory that a typical array would've required,
    # though it is 3 times faster in concatenations, it is rather slow in sorting,
    # thus the overall time consumption is similar.
    # buffer_size: GPU buffer size, MB.
    # down_sample_rate: Pixel sample ratio, i.e. pick one pixel every #down_sample_rate pixels.
    net.eval()
    buffer_size = buffer_size * 1024 * 1024 / 12  # MB -> how many pixels

    # 1 forward pass (sample predicted probabilities,
    # sorting here is unnecessary since there is relatively negligible time-consumption to consider)
    pseudo_label = torch.tensor([], dtype=torch.int64, device=device)
    pseudo_probability = torch.tensor([], dtype=torch.float32, device=device)
    probabilities = [np.array([], dtype=np.float32) for _ in range(num_classes)]
    with torch.no_grad():
        for images, _, heights, widths in tqdm(loader):
            images = images.to(device)
            outputs = net(images)['out']
            outputs = torch.nn.functional.interpolate(outputs, size=input_size, mode='bilinear', align_corners=True)

            # Generate pseudo labels (d1 x d2) and reassemble
            for i in range(0, len(heights)):
                prediction = crop(outputs[i], 0, 0, heights[i], widths[i])  # Back to the original size
                temp = prediction.softmax(dim=0)  # ! softmax
                temp = temp.max(dim=0)
                pseudo_label = torch.cat([pseudo_label, temp.indices.flatten()[:: down_sample_rate]])
                pseudo_probability = torch.cat([pseudo_probability, temp.values.flatten()[:: down_sample_rate]])

            # Count and reallocate
            if pseudo_probability.shape[0] > buffer_size:
                for j in range(num_classes):
                    probabilities[j] = np.concatenate((probabilities[j],
                                                       pseudo_probability[pseudo_label == j].cpu().numpy()))
                pseudo_label = torch.tensor([], dtype=torch.int64, device=device)
                pseudo_probability = torch.tensor([], dtype=torch.float32, device=device)

        # Final count
        for j in range(num_classes):
            probabilities[j] = np.concatenate((probabilities[j],
                                               pseudo_probability[pseudo_label == j].cpu().numpy()))

    # Sort (n * log(n) << n * label_ratio, so just sort is good) and find kc
    print('Sorting...')
    kc = []
    for j in range(num_classes):
        if len(probabilities[j]) == 0:
            with open('exceptions.txt', 'a') as f:
                f.write(str(time.asctime()) + '--' + str(j) + '\n')

    for j in tqdm(range(num_classes)):
        probabilities[j].sort()
        if label_ratio >= 1:
            kc.append(probabilities[j][0])
        else:
            if len(probabilities[j]) * label_ratio < 1:
                kc.append(0.00001)
            else:
                kc.append(probabilities[j][-int(len(probabilities[j]) * label_ratio) - 1])
    del probabilities  # Better be safe than...

    print(kc)
    return generate_pseudo_labels(net=net, device=device, loader=loader, cbst_thresholds=torch.tensor(kc),
                                  input_size=input_size, num_classes=num_classes)


# Test pseudo labels against ground truth labels (or whatever labels against whatever, depending on the DataSet object)
def test_labels(device, loader, categories, num_classes):
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for pred, target in tqdm(loader):
            pred, target = pred.to(device), target.to(device)
            conf_mat.update(target.flatten(), pred.flatten())

    acc_global, acc, iu = conf_mat.compute()
    print('Label status:')
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


def self_train_schedule(writer, labeled_loader, unlabeled_loader, pseudo_labeled_loader, validation_loader,
                        reference_loader, device, criterion, net, optimizer, lr_scheduler,
                        st_lr, init_epochs, st_epochs, num_iters, optimizer_init, is_mixed_precision,
                        init_label_ratio, det_label_ratio, categories, num_classes, input_sizes, exp_name,
                        val_num_steps=1000):
    # Conduct self-training iterations for semi-supervised & unsupervised training
    # Resume training from checkpoints is only supported for checkpoints made after each training round

    # Start with fully supervised baseline
    best_mIoU = train_one_iter(writer=writer, loader_c=labeled_loader, loader_sup=None,
                               validation_loader=validation_loader, net=net,
                               device=device, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
                               num_epochs=init_epochs, tensorboard_prefix='Supervised initial training ',
                               is_mixed_precision=is_mixed_precision, with_sup=False,
                               categories=categories, num_classes=num_classes, input_sizes=input_sizes)
    label_ratio = init_label_ratio

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
        # Pseudo labeling
        time_now = time.time()
        ratio = generate_class_balanced_pseudo_labels(net=net, device=device, loader=unlabeled_loader,
                                                      label_ratio=label_ratio,
                                                      input_size=input_sizes[0], num_classes=num_classes)
        print('Labeled ratio: %.2f' % ratio)
        writer.add_scalar('pseudo label ratio',
                          ratio,
                          i + 1)
        print('Pseudo labeling time: %.2fs' % (time.time() - time_now))

        # Test pseudo labels(Ground truth labels here are used for reporting this number only, no training
        # or hyper-parameter tuning in any way, shape or form is base on this)
        label_pixel_accuracy, label_mIoU = test_labels(device=device, loader=reference_loader,
                                                       categories=categories, num_classes=num_classes)
        writer.add_scalar('pseudo label pixel accuracy',
                          label_pixel_accuracy,
                          i + 1)
        writer.add_scalar('pseudo label mIoU',
                          label_mIoU,
                          i + 1)

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
                                   num_classes=num_classes, categories=categories, val_num_steps=val_num_steps)

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

        label_ratio += det_label_ratio

    return best_mIoU
