#!/bin/bash
CIFAR10_ROOT_DIR="../../../dataset/cifar10"
CIFAR10_DATA_DIR="../../../dataset/cifar10/cifar-10-batches-py"

echo downloading...
python download_cifar.py --dataset=cifar10 --base=${CIFAR10_ROOT_DIR}

echo Generating trainval/train/valtiny...
python generate_splits.py --seed=1 --dataset=cifar10 --base=${CIFAR10_DATA_DIR}

echo Renaming CIFAR10 test set...
cp ${CIFAR10_DATA_DIR}/test_batch ${CIFAR10_DATA_DIR}/test

echo Running 5 seeds on CIFAR10...
for seed in 1 2 3 4 5; do
    python generate_splits.py --seed=${seed} --dataset=cifar10 --base=${CIFAR10_DATA_DIR} --train-file=train_seed1
done

echo All done.
