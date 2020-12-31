import os
import math
import random
from utils.common import base_voc

random.seed(7777)

# Configurations
common_portion = 0.6  # Split the rest 50-50

sets_dir = os.path.join(base_voc, "ImageSets/Segmentation")
splits = [8, 20, 50, 106]

# sets_dir = os.path.join(base_city, "data_lists")
# splits = [2, 4, 8, 20, 30]

for i in range(3):
    for split in splits:
        ori_filename = str(split) + "_labeled_" + str(i) + ".txt"
        with open(os.path.join(sets_dir, ori_filename), "r") as f:
            lines = f.readlines()
        total_len = len(lines)
        random.shuffle(lines)

        # nc_len + nc_len + common_len = total_len
        nc_len = math.ceil((1 - common_portion) / 2 * total_len)
        common_len = total_len - 2 * nc_len
        split_index_l = nc_len + common_len
        split_index_r = nc_len
        lines_l = lines[:split_index_l]
        lines_r = lines[split_index_r:]
        print(str(len(lines_l)) + '--' + str(len(lines_r)) + ' :' + str(len(lines)))
        temp_filename = str(split) + "-l" + "_labeled_" + str(i) + ".txt"
        with open(os.path.join(sets_dir, temp_filename), "w") as f:
            f.writelines(lines_l)
        temp_filename = str(split) + "-r" + "_labeled_" + str(i) + ".txt"
        with open(os.path.join(sets_dir, temp_filename), "w") as f:
            f.writelines(lines_r)
