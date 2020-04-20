#!/bin/bash
echo 1/106
python main.py --exp-name=ss_voc_106-0 --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=106 --sets-id=0 --continue-from=fs_voc_106-0.pt --coco --mixed-precision
python main.py --exp-name=ss_voc_106-1 --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=106 --sets-id=1 --continue-from=fs_voc_106-1.pt --coco --mixed-precision
python main.py --exp-name=ss_voc_106-2 --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=106 --sets-id=2 --continue-from=fs_voc_106-2.pt --coco --mixed-precision
echo 1/106 valtiny
python main.py --exp-name=ss_voc_106-0_valtiny --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=106 --sets-id=0 --continue-from=fs_voc_106-0_valtiny.pt --coco --mixed-precision
python main.py --exp-name=ss_voc_106-1_valtiny --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=106 --sets-id=1 --continue-from=fs_voc_106-1_valtiny.pt --coco --mixed-precision
python main.py --exp-name=ss_voc_106-2_valtiny --gamma=9 --dataset=voc --val-num-steps=1000 --state=1 --iters=5 --init-epochs=0 --st-epochs 6 6 6 6 6 --st-lr=0.002 --split=106 --sets-id=2 --continue-from=fs_voc_106-2_valtiny.pt --coco --mixed-precision
echo Write the other shell scripts by yourself according to Readme.md!
echo Beware that log.txt only log valtiny perfromance when --valtiny is used.