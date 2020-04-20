#!/bin/bash
echo oracle
python main.py --exp-name=fs_voc_1 --dataset=voc --val-num-steps=1000 --state=2 --init-epochs=30 --split=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
echo 1/4
python main.py --exp-name=fs_voc_4-0 --dataset=voc --val-num-steps=500 --state=2 --init-epochs=60 --split=4 --sets-id=0 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_4-1 --dataset=voc --val-num-steps=500 --state=2 --init-epochs=60 --split=4 --sets-id=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_4-2 --dataset=voc --val-num-steps=500 --state=2 --init-epochs=60 --split=4 --sets-id=2 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
echo 1/8
python main.py --exp-name=fs_voc_8-0 --dataset=voc --val-num-steps=350 --state=2 --init-epochs=85 --split=8 --sets-id=0 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_8-1 --dataset=voc --val-num-steps=350 --state=2 --init-epochs=85 --split=8 --sets-id=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_8-2 --dataset=voc --val-num-steps=350 --state=2 --init-epochs=85 --split=8 --sets-id=2 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
echo 1/20
python main.py --exp-name=fs_voc_20-0 --dataset=voc --val-num-steps=220 --state=2 --init-epochs=134 --split=20 --sets-id=0 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_20-1 --dataset=voc --val-num-steps=220 --state=2 --init-epochs=134 --split=20 --sets-id=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_20-2 --dataset=voc --val-num-steps=220 --state=2 --init-epochs=134 --split=20 --sets-id=2 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
echo 1/50
python main.py --exp-name=fs_voc_50-0 --dataset=voc --val-num-steps=140 --state=2 --init-epochs=212 --split=50 --sets-id=0 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_50-1 --dataset=voc --val-num-steps=140 --state=2 --init-epochs=212 --split=50 --sets-id=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_50-2 --dataset=voc --val-num-steps=140 --state=2 --init-epochs=212 --split=50 --sets-id=2 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
echo 1/106
python main.py --exp-name=fs_voc_106-0 --dataset=voc --val-num-steps=100 --state=2 --init-epochs=300 --split=106 --sets-id=0 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_106-1 --dataset=voc --val-num-steps=100 --state=2 --init-epochs=300 --split=106 --sets-id=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
python main.py --exp-name=fs_voc_106-2 --dataset=voc --val-num-steps=100 --state=2 --init-epochs=300 --split=106 --sets-id=2 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision
echo 1/106 valtiny
python main.py --exp-name=fs_voc_106-0_valtiny --dataset=voc --val-num-steps=100 --state=2 --init-epochs=300 --split=106 --sets-id=0 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --valtiny
python main.py --exp-name=fs_voc_106-1_valtiny --dataset=voc --val-num-steps=100 --state=2 --init-epochs=300 --split=106 --sets-id=1 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --valtiny
python main.py --exp-name=fs_voc_106-2_valtiny --dataset=voc --val-num-steps=100 --state=2 --init-epochs=300 --split=106 --sets-id=2 --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --valtiny
echo Complete.
echo Check log.txt for final scores and check logs/ for tensorboard logs!
echo Beware that log.txt only log valtiny perfromance when --valtiny is used.
