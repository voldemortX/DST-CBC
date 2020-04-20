#!/bin/bash
echo OST
python main_ost.py --exp-name=ablations_b_ost --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision
echo OST-NA
python main_ost.py --exp-name=ablations_b_ost-na --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --no-aug
echo OST-T
python main_ost.py --exp-name=ablations_b_ost-t --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --threshold=0.9
echo OST-T-NA
python main_ost.py --exp-name=ablations_b_ost-t-na --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --threshold=0.9 --no-aug
echo ST
python main.py --exp-name=ablations_b_st --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --init-label-ratio=1
echo ST-NA
python main.py --exp-name=ablations_b_st-na --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --no-aug --init-label-ratio=1
echo Complete.
echo Check log.txt for final scores and check logs/ for tensorboard logs!
