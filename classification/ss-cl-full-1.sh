#!/bin/bash
seed="1"
split="100"
exp_name="cl_c10-100-1"
bs="512"
epoch="2120"
bs_l="16"
bs_p="496"

echo baseline
python main_fs.py --dataset=cifar10 --exp-name=${exp_name}__p0 --seed=${seed} --alpha=0.75 --weight-decay=0.0005 --ra --n=1 --m=10 --batch-size=${bs} --epochs=${epoch} --mixed-precision --train-set=${split}_seed${seed}_labeled
python main_fs.py --state=2 --continue-from=${exp_name}__p0--ema.pt

echo labeling
python main_dmt.py --labeling --train-set=${split}_seed${seed} --label-ratio=0.2 --continue-from=${exp_name}__p0--ema.pt

echo training
python main_dmt.py --mixed-precision --lr=0.1 --epochs=750 --batch-size-labeled=${bs_l} --batch-size-pseudo=${bs_p} --exp-name=${exp_name}__p1 --gamma1=0 --gamma2=0 --valtiny --fine-grain --weight-decay=0.0005 --seed=6 --n=1 --m=10 --start-at=375 --train-set=${split}_seed${seed}

echo labeling
python main_dmt.py --labeling --train-set=${split}_seed${seed} --label-ratio=0.4 --continue-from=${exp_name}__p1--ema.pt

echo training
python main_dmt.py --mixed-precision --lr=0.1 --epochs=750 --batch-size-labeled=${bs_l} --batch-size-pseudo=${bs_p} --exp-name=${exp_name}__p2 --gamma1=0 --gamma2=0 --valtiny --fine-grain --weight-decay=0.0005 --seed=7 --n=1 --m=10 --start-at=375 --train-set=${split}_seed${seed}

echo labeling
python main_dmt.py --labeling --train-set=${split}_seed${seed} --label-ratio=0.6 --continue-from=${exp_name}__p2--ema.pt

echo training
python main_dmt.py --mixed-precision --lr=0.1 --epochs=750 --batch-size-labeled=${bs_l} --batch-size-pseudo=${bs_p} --exp-name=${exp_name}__p3 --gamma1=0 --gamma2=0 --valtiny --fine-grain --weight-decay=0.0005 --seed=8 --n=1 --m=10 --start-at=375 --train-set=${split}_seed${seed}

echo labeling
python main_dmt.py --labeling --train-set=${split}_seed${seed} --label-ratio=0.8 --continue-from=${exp_name}__p3--ema.pt

echo training
python main_dmt.py --mixed-precision --lr=0.1 --epochs=750 --batch-size-labeled=${bs_l} --batch-size-pseudo=${bs_p} --exp-name=${exp_name}__p4 --gamma1=0 --gamma2=0 --valtiny --fine-grain --weight-decay=0.0005 --seed=9 --n=1 --m=10 --start-at=375 --train-set=${split}_seed${seed}

echo labeling
python main_dmt.py --labeling --train-set=${split}_seed${seed} --label-ratio=1 --continue-from=${exp_name}__p4--ema.pt

echo training
python main_dmt.py --mixed-precision --lr=0.1 --epochs=750 --batch-size-labeled=${bs_l} --batch-size-pseudo=${bs_p} --exp-name=${exp_name}__p5 --gamma1=0 --gamma2=0 --valtiny --fine-grain --weight-decay=0.0005 --seed=10 --n=1 --m=10 --start-at=375 --train-set=${split}_seed${seed}

echo testing
python main_fs.py --state=2 --continue-from=${exp_name}__p1--ema.pt
python main_fs.py --state=2 --continue-from=${exp_name}__p2--ema.pt
python main_fs.py --state=2 --continue-from=${exp_name}__p3--ema.pt
python main_fs.py --state=2 --continue-from=${exp_name}__p4--ema.pt
python main_fs.py --state=2 --continue-from=${exp_name}__p5--ema.pt

echo Done.
