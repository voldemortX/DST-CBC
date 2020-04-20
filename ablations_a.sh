#!/bin/bash
echo ST
python main.py --exp-name=ablations_a_st --gamma=0 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --init-label-ratio=1
echo DST
python main.py --exp-name=ablations_a_dst --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=1 --init-epochs=0 --st-epochs 30 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision --init-label-ratio=1
echo DST-C
python main_dstc.py --exp-name=ablations_a_dst-c --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision
echo DST-CBC
python main.py --exp-name=ablations_a_dst-cbc --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision
echo OW
python main_ow.py --exp-name=ablations_a_ow --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=20 --sets-id=1 --continue-from=fs_voc_20-1.pt --coco --mixed-precision
echo Complete.
echo Check log.txt for final scores and check logs/ for tensorboard logs!
