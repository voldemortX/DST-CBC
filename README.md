# Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum

Untested bug fix for random scale.

## Main results

Untested.

## Preparations
You'll need a CUDA 10, Python3 enviroment (best on Linux) with PyTorch 1.2.0, TorchVision 0.4.0 and Apex to run the code in this repo.

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
chmod 777 *.sh
```

### 4. Download and convert the pre-trained model:

Pytorch automitically downloads the ImageNet pre-trained models, for the exact COCO-pretrained model used in previous work [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg):

```
./prepare_coco.sh
```

### 5. Prepare the datasets:

The PASCAL VOC 2012 dataset we used is the commonly used 10582 training set version. If you don't already have that dataset, we refer you to [Google](https://www.google.com) or this [blog](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/).

The Cityscapes dataset can be downloaded in their [official website](https://www.cityscapes-dataset.com/).

When you have done all above procedures and got the datasets, you also need to change the base directories [here](https://github.com/voldemortX/DST-CBC/blob/master/data_processing.py#L7). Then prepare the Cityscapes dataset:

```
python cityscapes_data_lists.py
```

### 6. Prepare the data splits used in the paper:

We already provided the exact data splits, including *valtiny* [here](https://github.com/voldemortX/DST-CBC/tree/master/data_splits), if you use Python 3.6, you should be getting the same data splits using *generate_splits.py*. The data splits for PASCAL VOC 2012 need to be placed at: 

```
your_data_dir/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation
```

The data splits for Cityscapes need to be placed at:

```
your_data_dir/data_lists
```

Create that directory if you don't have it.

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

## Run the code
The command line arguments are diversified to meet every possible needs. We provide examples in scripts. Final results can be found at log.txt after training.

To train fully-supervised baselines on all data splits:

```
./fs_voc_all.sh
./fs_city_all.sh
```

To conduct semi-supervised learning with DST-CBC after you've acquired the baselines (writing all of the commands would be too long, we only give examples on the samllest labeled ratio of PASCAL VOC):
```
./ss_voc_example.sh
```

To visualize the training process with tensorboard:

```
tensorboard --logdir=logs
```

To evaluate a model (e.g. might need evaluation on *val* when trained with *valtiny*):

```
python main.py --state=3 --dataset=voc/city --continue-from=your_model.pt --mixed-precision --coco
```

## Understand the code
We refer interested readers to this repository's [wiki](https://github.com/voldemortX/DST-CBC/wiki).

## Acknowledgements

The DeepLabV2 network architecture is faithfully re-implemented from [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg).

The CBC part of the code is adapted from [CRST](https://github.com/yzou2/CRST).

And the overall implementation is based on [TorchVision](https://github.com/pytorch/vision) and [PyTorch](https://github.com/pytorch/pytorch).

## Notes
It's best to use a **Turing** or **Volta** architecture GPU when running our code, since they have tensor cores and the computation speed is much faster with mixed precision. For instance, RTX 2080 Ti (which is what we used) or Tesla V100.

Our implementation is fast and memory efficient. A whole run (5 rounds of DST-CBC on PASCAL VOC 2012) takes about 7 hours on a single RTX 2080 Ti using up to 6GB graphic memory, including on-the-fly evaluations. The Cityscapes experiments are even faster.

## Contact
If you have any questions that are not answerable with Google, feel free to contact us through zyfeng97@outlook.com.

Issues and PRs are also welcomed. 

## Citation

```
@article{feng2020semi,
  title={Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum},
  author={Feng, Zhengyang and Zhou, Qianyu and Cheng, Guangliang and Tan, Xin and Shi, Jianping and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2004.08514},
  year={2020}
}
```
