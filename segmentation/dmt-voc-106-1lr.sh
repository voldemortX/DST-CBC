#!/bin/bash

# lr: left-right, i.e. 2 different samplings of the labeled subset
train_set="106"
sid="1"
exp_name="dmt-voc-106-1lr"
ep="4"
lr="0.001"
c1="5"
c2="5"
i1="5"
i2="5"
phases=("1" "2" "3" "4" "5")
rates=("0.2" "0.4" "0.6" "0.8" "1")

python main.py --exp-name=${exp_name}__p0--l --val-num-steps=100 --state=2 --epochs=300 --train-set=${train_set}-l --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=1
python main.py --exp-name=${exp_name}__p0--r --val-num-steps=100 --state=2 --epochs=300 --train-set=${train_set}-r --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=2

echo dmt
for i in ${!rates[@]}; do
  echo ${phases[$i]}--${rates[$i]}
  
  echo labeling
  python main.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

  echo training
  python main.py --exp-name=${exp_name}__p${phases[$i]}--r --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --batch-size-labeled=1 --batch-size-pseudo=7 --seed=1
  
  echo labeling
  python main.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

  echo training
  python main.py --exp-name=${exp_name}__p${phases[$i]}--l --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision --epochs=${ep} --gamma1=${i1} --gamma2=${i2} --lr=${lr} --batch-size-labeled=1 --batch-size-pseudo=7 --seed=2
        
done
