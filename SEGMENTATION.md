# DMT for Semantic Segmentation

This folder contains code for the semantic segmentation experiments for Dynamic Mutual Training (DMT). Supported datasets include PASCAL VOC 2012 and Cityscapes.

## Results

### Semi-supervised semantic segmentation *mIOU* on PASCAL VOC 2012 *val*:


method | network | 1/106<br>(100 labels) | 1/50 | 1/20 | 1/8 | full<br>(Oracle) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
Baseline | DeepLab-v2 | 46.66 | 55.62 | 62.29 | 67.37 | 74.75 |
Mean Teacher | DeepLab-v2 | - | - | - | 67.65 | 73.59 |
Hung et al. | DeepLab-v2 | - | 57.2 | 64.7 | 69.5 | 74.9 |
s4GAN + MLMT | DeepLab-v2 | - | 63.3 | 67.2 | 71.4 | **75.6** |
CutMix | DeepLab-v2 | 53.79 | 64.81 | 66.48 | 67.60 | 72.54 |
CCT | PSPNet | - | - | -	| 70.45 | 75.25 |
GCT | DeepLab-v2 | - | - | -	| 70.57 | 74.06 |
DMT | DeepLab-v2 | **63.04** | **67.15** | **69.92** | **72.70** | 74.75 |

### Semi-supervised semantic segmentation *mIoU* on Cityscapes *val*:

method | network | 1/30<br>(100 labels) | 1/8 | full<br>Oracle |
|:--:|:--:|:--:|:--:|:--:|
Baseline | DeepLab-v2 | 49.54 | 59.65 | **68.16** |
Hung et al. | DeepLab-v2 |  - | 58.8 | 67.7 |
s4GAN + MLMT | DeepLab-v2 | - | 59.3 | 65.8 |
CutMix | DeepLab-v2 | 51.20 | 60.34 | 67.68 |
DMT | DeepLab-v2 | **54.80** | **63.03** | **68.16** |

### DMT against SBD annotations on PASCAL VOC 2012 *val*:

| method | mIoU (%) |
|:--:|:--:|
| PASCAL VOC 1464 labels | 72.10 |
| PASCAL VOC 1464 labels + SBD 9118 labels | 74.75 |
| PASCAL VOC 1464 labels + DMT | **74.85** |

## Preparations

### 1. Set dataset paths:

Set the directories of your datasets [here](https://github.com/voldemortX/DMT/blob/master/segmentation/utils/common.py#L10).

### 2. Download and process the dataset:

The PASCAL VOC 2012 dataset we used is the commonly used 10582 training set version. If you don't already have that dataset, we refer you to [Google](https://www.google.com) or this [blog](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/).

The Cityscapes dataset can be downloaded in their [official website](https://www.cityscapes-dataset.com/).

When you have done all above procedures and got the datasets, you can prepare the Cityscapes dataset:

```
python cityscapes_data_list.py
```

Then generate 3 different random splits:

```
python generate_splits.py
```

You can check your generated splits against our uploaded splits in the older [DST-CBC](https://github.com/voldemortX/DST-CBC/tree/dst-cbc) branch, also you can download the *valtiny* set there.

Afterwards, your data directory structure should look like these:

    ├── your_voc_base_dir/VOCtrainval_11-May-2012/VOCdevkit/VOC2012                    
        ├── Annotations 
        ├── ImageSets
        │   ├── Segmentation
        │   │   ├── 1_labeled_0.txt
        │   │   ├── 1_labeled_1.txt
        │   │   └── ... 
        │   └── ... 
        ├── JPEGImages
        ├── SegmentationClass
        ├── SegmentationClassAug
        └── ...

    ├── your_city_base_dir                     
        ├── data_lists
        │   ├── 1_labeled_0.txt
        │   ├── 1_labeled_1.txt
        │   └── ...  
        ├── gtFine
        └── leftImage8bit

### 3. Prepare pre-trained weights:

For segmentation experiments, we need COCO pre-trained weights same as previous works:

```
./prepare_coco.sh
```

ImageNet pre-trained weights will be automatically downloaded when running code.

**\[Note\]** If you can't download the COCO weights, try the [Google Drive link](https://drive.google.com/file/d/14j4RoLqnfGeKaBN7m11u8XUnuov6PgYp/view?usp=sharing).

## Run the code

For multi-GPU/TPU/Distributed machine users, first run:

```
accelerate config
```

More details can be found at [Accelerate](https://github.com/huggingface/accelerate). Note that the mixed precision config cannot be used, you should still use `--mixed-precision` for that.

We provide examples in scripts and commands. Final results can be found at log.txt after training.

For example, run DMT with different pre-trained weights:

```
./dmt-voc-20-1.sh
```

Run DMT with difference maximized sampling:

```
python dms_sample.py
./dmt-voc-106-1lr.sh
```

Run the surpassing human supervision experiment:

```
python pascal_sbd_split.py
./dmt-voc-sbd-0.sh
```


Of course you'll need to run 3 times average to determine performance by changing the `sid` parameter (we used 0,1,2) in shell scripts.

We also provide scripts for ablations, be sure to run *abl_baseline.sh* first. 

For small validation set, use `--valtiny`.
