# DMT for Image Classification

This folder contains code for the CIFAR-10 image classification experiments for Dynamic Mutual Training (DMT).

## Results

### Semi-supervised learning accuracy on CIFAR-10 *test*:

| method | network | 4000 labels |
|:--:|:--:|:--:|
| Baseline | WRN-28-2 | 86.08 |
| Mean Teacher | WRN-28-2 | 89.64 |
| DCT (two models) | CNN-13 | 90.97 |
| Dual Student | CNN-13 | 91.11 |
| MixMatch | WRN-28-2 | 93.76 |
| DAG | CNN-13 | 93.87 |
| Curriculum labeling | WRN-28-2 | 94.02 |
| DMT | WRN-28-2 | **94.21** |

## Preparations

### 1. Set dataset paths:

Set the directory that you want your dataset at [here](https://github.com/voldemortX/DST-CBC/blob/master/classification/generate_splits.sh#L2) and [here](https://github.com/voldemortX/DST-CBC/blob/master/classification/utils/common.py#L12).

### 2. Download and process the dataset:

The CIFAR-10 dataset can be downloaded and splitted to 5 random splits and validation set (200 image small validation set) by:

```
./generate_splits.sh
```


## Run the code

For multi-GPU/TPU/Distributed machine users, first run:

```
accelerate config
```

More details can be found at [Accelerate](https://github.com/huggingface/accelerate). Note that the mixed precision config cannot be used, you should still use `--mixed-precision` for that.

We provide examples in scripts and commands. Final results can be found at `log.txt` after training.

For example, with 1000 labels, to compare CL and DMT in a controlled experiment with same baseline model to start training:

```
./ss-cl-full-1.sh
./ss-dmt-full-1.sh
```

You'll need to run 5 times average to determine performance by changing the `seed` parameter (we used 1,2,3,4,5) in shell scripts.

For small validation set, use `--valtiny`; for fine-grained testing, use `--fine-grain`.
