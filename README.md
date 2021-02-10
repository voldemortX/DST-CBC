# DMT: Dynamic Mutual Training for Semi-Supervised Learning

This repository contains the code for our paper [DMT: Dynamic Mutual Training for Semi-Supervised Learning](https://arxiv.org/abs/2004.08514), a concise and effective method for semi-supervised semantic segmentation & image classification.

Some might know it as the previous version **DST-CBC**, or *Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum*, if you want the old code, you can check out the [dst-cbc](https://github.com/voldemortX/DST-CBC/tree/dst-cbc) branch.

<div align="center">
  <img src="overview.png"/>
</div>

## News

### 2021.2.10

A slight backbone architecture difference in the segmentation task has just been identified and described in Acknowledgement.

### 2021.1.1

DMT is released. Happy new year! :wink: 

### 2020.12.7

The bug fix for DST-CBC (not fully tested) is released at the [scale](https://github.com/voldemortX/DST-CBC/tree/scale) branch.

### 2020.11.9

~~Stay tuned for Dynamic Mutual Training (DMT), an updated version of DST-CBC, which has overall better and stabler performance and will be released early November.~~
**A new version Dynamic Mutual Training (DMT) will be released later, which has overall better and stabler performance.**

Also, thanks to [**@lorenmt**](https://github.com/lorenmt), a data augmentation bug fix will be released along with the next version, where PASCAL VOC performance is overall boosted by 1~2%, Cityscapes could also have better performance. But probably the gap to oracle will remain similar.

## Setup

You'll need a CUDA 10, Python3 environment (best on Linux) with PyTorch 1.2.0, TorchVision 0.4.0 and Apex to run the code in this repo.

### 1. Setup the exact version of Apex & PyTorch & TorchVision for mixed precision training:

```
pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl && pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
!There seems to be an issue of apex installations from the official repo sometimes. If you encounter errors, we suggest you use our stored older apex [codes](https://drive.google.com/open?id=1x8enpvdTTZ3RChf17XvcLdSYulUPg3sR).

**PyTorch 1.6** now includes automatic mixed precision at apex level "O1". We probably will update this repo accordingly in the future. 

### 2. Install other python packages you may require:

collections, future, matplotlib, numpy, PIL, shutil, tensorboard, tqdm

### 3. Download the code and prepare the scripts:

```
git clone https://github.com/voldemortX/DST-CBC.git
cd DST-CBC
chmod 777 segmentation/*.sh
chmod 777 classification/*.sh
```

## Getting started

Get started with [SEGMENTATION.md](SEGMENTATION.md) for semantic segmentation.

Get started with [CLASSIFICATION.md](CLASSIFICATION.md) for image classification.

## Understand the code
We refer interested readers to this repository's [wiki](https://github.com/voldemortX/DST-CBC/wiki). *It is not updated for DMT yet.*

## Notes
It's best to use a **Turing** or **Volta** architecture GPU when running our code, since they have tensor cores and the computation speed is much faster with mixed precision. For instance, RTX 2080 Ti (which is what we used) or Tesla V100, RTX 20/30 series.

Our implementation is fast and memory efficient. A whole run (train 2 models by DMT on PASCAL VOC 2012) takes about 8 hours on a single RTX 2080 Ti using up to 6GB graphic memory, including on-the-fly evaluations and training baselines. The Cityscapes experiments are even faster.

## Contact

Issues and PRs are most welcomed. 

If you have any questions that are not answerable with Google, feel free to contact us through zyfeng97@outlook.com.

## Citation

```
@article{feng2020dmt,
  title={DMT: Dynamic Mutual Training for Semi-Supervised Learning},
  author={Feng, Zhengyang and Zhou, Qianyu and Gu, Qiqi and Tan, Xin and Cheng, Guangliang and Lu, Xuequan and Shi, Jianping and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2004.08514},
  year={2020}
}
```

## Acknowledgements

The DeepLabV2 network architecture and coco pre-trained weights are faithfully re-implemented from [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg). The only difference is we use the so-called ResNetV1.5 implementation for ResNet-101 backbone (same as torchvision), for difference between ResNetV1 and V1.5, refer to [this issue](https://github.com/pytorch/vision/issues/191).

The CBC part of the older version DST-CBC is adapted from [CRST](https://github.com/yzou2/CRST).

The overall implementation is based on [TorchVision](https://github.com/pytorch/vision) and [PyTorch](https://github.com/pytorch/pytorch).

