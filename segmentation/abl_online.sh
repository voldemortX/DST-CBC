#!/bin/bash

train_set="50"
sid="2"
exp_name="abl_online"
ep="20"
lr="0.001"

echo online
python main_online.py --exp-name=${exp_name} --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p0--c.pt --coco --mixed-precision --epochs=${ep} --lr=${lr} --seed=1
