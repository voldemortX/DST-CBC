import random
import os
from utils.common import base_city, base_voc


# Configurations (! Change the filenames in float, e.g. 29.75)

# sets_dir = os.path.join(base_voc, "ImageSets/Segmentation")
# whole_train_set = "trainaug.txt"
# splits = [2, 4, 8, 20, 105.82]

sets_dir = os.path.join(base_city, "data_lists")
whole_train_set = "train.txt"
splits = [2, 4, 8, 20, 29.75]

random.seed(7777)

# Open original file
with open(os.path.join(sets_dir, whole_train_set), "r") as f:
    file_names = f.readlines()
original_train_size = len(file_names)
print("Original training set size: " + str(original_train_size))

# ! Check for line EOF
if '\n' not in file_names[original_train_size - 1]:
    file_names[original_train_size - 1] += "\n"

# 3 random splits
# So as to guarantee smaller sets are included in bigger sets
for i in range(3):
    random.shuffle(file_names)

    # Semi-supervised splits
    for split in splits:
        split_index = int(original_train_size / split)  # Floor
        if split_index % 8 == 1:  # For usual batch-size (avoid BN problems)
            split_index += 1
        with open(os.path.join(sets_dir, str(split) + "_labeled_" + str(i) + ".txt"), "w") as f:
            f.writelines(file_names[0: split_index])
        with open(os.path.join(sets_dir, str(split) + "_unlabeled_" + str(i) + ".txt"), "w") as f:
            f.writelines(file_names[split_index:])

    # Whole set (fully-supervised & fully-unsupervised), to be consistent in naming
    with open(os.path.join(sets_dir, "1_labeled_" + str(i) + ".txt"), "w") as f:
        f.writelines(file_names)
    with open(os.path.join(sets_dir, "0_unlabeled_" + str(i) + ".txt"), "w") as f:
        f.writelines(file_names)
