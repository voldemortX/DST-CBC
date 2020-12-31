#!/bin/bash

train_set="50"
sid="2"
exp_name="abl_flip"
ep="4"
lr="0.001"
c1="5"
c2="5"
i1="5"
i2="5"
phases=("1" "2" "3" "4" "5")
rates=("0.2" "0.4" "0.6" "0.8" "1")

echo flip
for i in ${!rates[@]}; do
  echo ${phases[$i]}--${rates[$i]}
  
  echo labeling
  python main_flip.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

  echo training
  python main_flip.py --exp-name=${exp_name}__p${phases[$i]}--r --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --seed=1
  
  echo labeling
  python main_flip.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

  echo training
  python main_flip.py --exp-name=${exp_name}__p${phases[$i]}--l --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --seed=2
        
done
