#!/bin/bash
echo oracle
python main.py --exp-name=fs_city_1 --dataset=city --val-num-steps=1000 --state=2 --init-epochs=60 --split=1 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
echo 1/4
python main.py --exp-name=fs_city_4-0 --dataset=city --val-num-steps=500 --state=2 --init-epochs=120 --split=4 --sets-id=0 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
python main.py --exp-name=fs_city_4-1 --dataset=city --val-num-steps=500 --state=2 --init-epochs=120 --split=4 --sets-id=1 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
python main.py --exp-name=fs_city_4-2 --dataset=city --val-num-steps=500 --state=2 --init-epochs=120 --split=4 --sets-id=2 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
echo 1/8
python main.py --exp-name=fs_city_8-0 --dataset=city --val-num-steps=350 --state=2 --init-epochs=170 --split=8 --sets-id=0 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
python main.py --exp-name=fs_city_8-1 --dataset=city --val-num-steps=350 --state=2 --init-epochs=170 --split=8 --sets-id=1 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
python main.py --exp-name=fs_city_8-2 --dataset=city --val-num-steps=350 --state=2 --init-epochs=170 --split=8 --sets-id=2 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
echo 1/30
python main.py --exp-name=fs_city_30-0 --dataset=city --val-num-steps=100 --state=2 --init-epochs=330 --split=30 --sets-id=0 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
python main.py --exp-name=fs_city_30-1 --dataset=city --val-num-steps=100 --state=2 --init-epochs=330 --split=30 --sets-id=1 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
python main.py --exp-name=fs_city_30-2 --dataset=city --val-num-steps=100 --state=2 --init-epochs=330 --split=30 --sets-id=2 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --lr=0.004
echo 1/30 valtiny
python main.py --exp-name=fs_city_30-0_valtiny --dataset=city --val-num-steps=100 --state=2 --init-epochs=330 --split=30 --sets-id=0 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --valtiny --lr=0.004
python main.py --exp-name=fs_city_30-1_valtiny --dataset=city --val-num-steps=100 --state=2 --init-epochs=330 --split=30 --sets-id=1 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --valtiny --lr=0.004
python main.py --exp-name=fs_city_30-2_valtiny --dataset=city --val-num-steps=100 --state=2 --init-epochs=330 --split=30 --sets-id=2 --continue-from=city_coco_resnet101.pt --coco --mixed-precision --valtiny --lr=0.004
echo Complete.
echo Check log.txt for final scores and check logs/ for tensorboard logs!
echo Beware that log.txt only log valtiny perfromance when --valtiny is used.
