import os
from utils.common import base_voc

sets_dir = os.path.join(base_voc, "ImageSets/Segmentation")
whole_train_set = "trainaug.txt"
pascal_train_set = 'train.txt'


def open_file(dir_name, train_set):
    # Open files
    with open(os.path.join(dir_name, train_set), "r") as f:
        file_names = f.readlines()
    train_size = len(file_names)
    print("Set size: " + str(train_size))

    # ! Check for line EOF
    if '\n' not in file_names[train_size - 1]:
        file_names[train_size - 1] += "\n"

    return file_names


whole = open_file(sets_dir, whole_train_set)
pascal = open_file(sets_dir, pascal_train_set)
sbd = []
for x in whole:
    if x not in pascal:
        sbd.append(x)

assert len(sbd) == 9118
with open(os.path.join(sets_dir, 'train_labeled_0.txt'), "w") as f:
    f.writelines(pascal)
with open(os.path.join(sets_dir, 'train_unlabeled_0.txt'), "w") as f:
    f.writelines(sbd)
print('Done.')
