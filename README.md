# DATR: Unsupervised Domain Adaptive Detection Transformer with Dataset-Level Adaptation and Prototypical Alignment

By Jianhong Han, Liang Chen and Yupei Wang.

This repository contains the implementation accompanying our paper DATR: Unsupervised Domain Adaptive Detection Transformer with Dataset-Level Adaptation and Prototypical Alignment.

If you find it helpful for your research, please consider citing:

```
@ARTICLE{10841964,
  author={Chen, Liang and Han, Jianhong and Wang, Yupei},
  journal={IEEE Transactions on Image Processing}, 
  title={DATR: Unsupervised Domain Adaptive Detection Transformer With Dataset-Level Adaptation and Prototypical Alignment}, 
  year={2025},
  volume={34},
  pages={982-994}}

```


![](/figs/Figure1.png)

## Acknowledgment
This implementation is bulit upon [DINO](https://github.com/IDEA-Research/DINO/) and [RemoteSensingTeacher](https://github.com/h751410234/RemoteSensingTeacher).

## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 16.04
* Python: 3.10.9
* CUDA: 11.8
* PyTorch: 2.0.1 (The lower versions of Torch can cause some bugs.)
* torchvision: 0.15.2

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [DAcoco.py](./datasets/DAcoco.py)

```
  #---源域
    PATHS_Source = {
        "train": ("",  #train image dir
                  ""), #train coco format json file
        "val": ("",    #val image dir
                ""),   #val coco format json file
    }
    #----目标域
    PATHS_Target = {
        "train": ("",  #train image dir
                  ""), #train coco format json file
        "val": ("",    #val image dir
                ""),   #val coco format json file
    }
```
- Add domain adaptation direction within the script [__init__.py](./datasets/__init__.py). For example:
```
    if args.dataset_file == 'city':
        return build_city_DA(image_set, args,strong_aug)
```

## Training / Evaluation / Inference
We provide training script as follows.
We divide the training process into two stages. The settings for each stage can be found in the config folder.

(1) For the Burn-In stage:
- Training with single GPU
```
sh scripts/DINO_train.sh
```
- Training with Multi-GPU
```
sh scripts/DINO_train_dist.sh
```
(2) For the Teacher-Student Mutual Learning stage, it is necessary to use the optimal model obtained from the first stage of training.
- Training with single GPU
```
sh scripts/DINO_train_self_training.sh
```
- Training with Multi-GPU
```
sh scripts/DINO_train_self_training_dist.sh
```

We provide evaluation script to evaluate pre-trained model. 
- Evaluation Model.
```
sh scripts/DINO_eval.sh
```
- Evaluation EMA Model.
```
sh scripts/DINO_eval_for_EMAmodel.sh
```

We provide inference script to visualize detection results. See [inference.py](inference.py) for details
- Inference Model.
```
python inference.py
```
- Inference EMA Model.
```
python inference_ema_model.py 
```
## Pre-trained models
We provide specific experimental configurations and pre-trained models to facilitate the reproduction of our results. 
You can learn the details of DATR through the paper, and please cite our papers if the code is useful for your papers. Thank you!

Task | mAP50  | Config | Model 
------------| ------------- | -------------| -------------
**Cityscapes to Foggy Cityscapes**  | 52.8% | [cfg](config/DA/Cityscapes2FoggyCityscapes) | [model](https://pan.baidu.com/s/1ZGvYjwXUMoBqtcGHfTnNww?pwd=mxxg)
**Sim10k to Cityscapes** | 66.3% | [cfg](config/DA/Sim10k2Cityscapes) | [model](https://pan.baidu.com/s/17ZS4IsFxAeyfVessQnHvRA?pwd=mw9u)
**Cityscapes to BDD100K-daytime** | 41.9% | [cfg](config/DA/Cityscapes2BDD100k) | [model](https://pan.baidu.com/s/17UElPN8gdqd7paE0B149Vg?pwd=dwes)

## Reference
https://github.com/IDEA-Research/DINO

https://github.com/h751410234/RemoteSensingTeacher