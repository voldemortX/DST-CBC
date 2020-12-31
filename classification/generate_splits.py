import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm


# Configs
cifar10_dict = {
    'list': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
    'sizes': [4, 25, 100, 400],  # per-class
    'num_classes': 10,
    'val_size': 20  # per-class
}


def extract(base, lists):
    # Original format: {'data': np.arr(n*3072, uint8),
    #                   'labels/fine_labels': [int, int, ...],
    #                   ...}
    # The generate file sizes are smaller because original version has redundant keys
    data = []
    targets = []
    for file_name in lists:
        file_path = os.path.join(base, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])

    data = np.vstack(data)
    targets = np.array(targets)
    return data, targets


def shuffle(data, targets, sizes, num_classes, base, seed):
    print('Shuffling...')
    np.random.seed(seed)
    final_data_labeled = [None for _ in range(len(sizes))]
    final_targets_labeled = [[] for _ in range(len(sizes))]
    final_data_unlabeled = [None for _ in range(len(sizes))]
    final_targets_unlabeled = [[] for _ in range(len(sizes))]

    for i in tqdm(range(num_classes)):
        # Shuffle by class (as done by other methods)
        total = data[targets == i]
        np.random.shuffle(total)

        # Assured superiority on training sets (i.e. larger sets include smaller sets)
        for j in range(len(sizes)):
            if final_data_labeled[j] is None:
                final_data_labeled[j] = total[:sizes[j], :]
            else:
                final_data_labeled[j] = np.concatenate((final_data_labeled[j], total[:sizes[j], :]))
            if final_data_unlabeled[j] is None:
                final_data_unlabeled[j] = total[sizes[j]:, :]
            else:
                final_data_unlabeled[j] = np.concatenate((final_data_unlabeled[j], total[sizes[j]:, :]))
            final_targets_labeled[j] += [i for _ in range(sizes[j])]
            final_targets_unlabeled[j] += [i for _ in range(total.shape[0] - sizes[j])]
            assert final_data_unlabeled[j].shape[0] == len(final_targets_unlabeled[j])
            assert final_data_labeled[j].shape[0] == len(final_targets_labeled[j])
            assert final_data_labeled[j].shape[0] / sizes[j] == (i + 1)

    # Store
    print('Saving splits...')
    for i in tqdm(range(len(sizes))):
        with open(os.path.join(base, str(sizes[i]) + '_seed' + str(seed) + '_labeled'), 'wb') as f:
            pickle.dump({'data': final_data_labeled[i], 'labels': final_targets_labeled[i]}, f)
        with open(os.path.join(base, str(sizes[i]) + '_seed' + str(seed) + '_unlabeled'), 'wb') as f:
            pickle.dump({'data': final_data_unlabeled[i], 'labels': final_targets_unlabeled[i]}, f)


def train_val_split(data, targets, val_size, num_classes, base, seed):
    # Split train & valtiny and store a full trainval as well
    print('Saving trainval...')
    assert data.shape[0] == len(targets)
    with open(os.path.join(base, 'trainval_seed' + str(seed)), 'wb') as f:
        pickle.dump({'data': data, 'labels': targets.tolist()}, f)

    print('Shuffling...')
    final_data_valtiny = None
    final_data_train = None
    final_targets_valtiny = []
    final_targets_train = []
    for i in tqdm(range(num_classes)):
        # Shuffle by class (as done by other methods)
        total = data[targets == i]
        np.random.shuffle(total)
        if final_data_train is None:
            final_data_train = total[:-val_size, :]
        else:
            final_data_train = np.concatenate((final_data_train, total[:-val_size, :]))
        if final_data_valtiny is None:
            final_data_valtiny = total[-val_size:, :]
        else:
            final_data_valtiny = np.concatenate((final_data_valtiny, total[-val_size:, :]))
        final_targets_train += [i for _ in range(total.shape[0] - val_size)]
        final_targets_valtiny += [i for _ in range(val_size)]

    print('Saving train & valtiny...')
    assert final_data_train.shape[0] == len(final_targets_train)
    assert final_data_valtiny.shape[0] == len(final_targets_valtiny)
    with open(os.path.join(base, 'train_seed' + str(seed)), 'wb') as f:
        pickle.dump({'data': final_data_train, 'labels': final_targets_train}, f)
    with open(os.path.join(base, 'valtiny_seed' + str(seed)), 'wb') as f:
        pickle.dump({'data': final_data_valtiny, 'labels': final_targets_valtiny}, f)


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='Generator')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Cifar10 (default: cifar10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--base', type=str, default='../',
                        help='Dataset directory (default: ../)')
    parser.add_argument('--train-file', type=str, default=None,
                        help='File for the training set, w.o. val, None means doing train-val split (default: None)')
    args = parser.parse_args()
    if args.dataset == 'cifar10':
        param_dict = cifar10_dict
    else:
        raise NotImplementedError
    if args.train_file is None:  # Split original data to trainval/train/valtiny
        data, targets = extract(base=args.base, lists=param_dict['list'])
        train_val_split(data=data, targets=targets, val_size=param_dict['val_size'],
                        num_classes=param_dict['num_classes'], base=args.base, seed=args.seed)
    else:  # Split train set to labeled & unlabeled
        data, targets = extract(base=args.base, lists=[args.train_file])
        shuffle(data=data, targets=targets, sizes=param_dict['sizes'], num_classes=param_dict['num_classes'],
                base=args.base, seed=args.seed)

    print('Splits done. Check ' + args.base)
