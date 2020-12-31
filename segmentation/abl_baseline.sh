#!/bin/bash

train_set="50"
sid="2"
exp_name="abl_baseline"

python main.py --exp-name=${exp_name}__p0--c --val-num-steps=220 --state=2 --epochs=134 --train-set=${train_set} --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --seed=1
python main.py --exp-name=${exp_name}__p0--l --val-num-steps=220 --state=2 --epochs=134 --train-set=${train_set}-l --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --seed=1
python main.py --exp-name=${exp_name}__p0--r --val-num-steps=220 --state=2 --epochs=134 --train-set=${train_set}-r --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --seed=2

echo copying
cp ${exp_name}__p0--c.pt abl_online__p0--c.pt
cp ${exp_name}__p0--c.pt abl_cbst__p0--c.pt
cp ${exp_name}__p0--c.pt abl_dst__p0--c.pt
cp ${exp_name}__p0--l.pt abl_naive__p0--l.pt
cp ${exp_name}__p0--r.pt abl_naive__p0--r.pt
cp ${exp_name}__p0--l.pt abl_flip__p0--l.pt
cp ${exp_name}__p0--r.pt abl_flip__p0--r.pt
