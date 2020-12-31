#!/bin/bash

train_set="train"
sid="0"
exp_name="dmt-voc-sbd-0"
ep="5"
lr="0.001"
c1="5"
c2="5"
i1="5"
i2="5"
s1="1"
s2="2"
phases=("1" "2" "3" "4" "5")
rates=("0.2" "0.4" "0.6" "0.8" "1")

echo fs
python main.py --exp-name=${exp_name}__oracle --state=2 --epochs=30 --train-set=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=${sid}

python main.py --exp-name=${exp_name}__p0--c --val-num-steps=350 --state=2 --epochs=85 --train-set=${train_set} --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=${s1}
python main.py --exp-name=${exp_name}__p0--i --val-num-steps=350 --state=2 --epochs=85 --train-set=${train_set} --sets-id=${sid} --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=${s2}

echo dmt
for i in ${!rates[@]}; do
  echo ${phases[$i]}--${rates[$i]}
  
  echo labeling
  python main.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--i.pt --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

  echo training
  python main.py --exp-name=${exp_name}__p${phases[$i]}--c --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--c.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --batch-size-pseudo=7 --batch-size-labeled=1 --seed=${s1} 
  
  echo labeling
  python main.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--c.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

  echo training
  python main.py --exp-name=${exp_name}__p${phases[$i]}--i --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--i.pt --mixed-precision --epochs=${ep} --gamma1=${i1} --gamma2=${i2} --lr=${lr} --batch-size-pseudo=7 --batch-size-labeled=1 --seed=${s2}
        
done
