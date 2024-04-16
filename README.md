# DATR: Unsupervised Domain Adaptive Detection Transformer with Prototypical Adaptation

By Jianhong Han, Liang Chen and Yupei Wang.

This repository contains the implementation accompanying our paper DATR: Unsupervised Domain Adaptive Detection Transformer with Prototypical Adaptation.

<!--
#--暂时注释
If you find it helpful for your research, please consider citing:

```
@inproceedings{XXX,
  title     = {Remote Sensing Teacher: Cross-Domain Detection Transformer with Learnable Frequency-Enhanced Feature Alignment in Remote Sensing Imagery},
  author    = {Jianhong Han, Yupei Wang, Liang Chen},
  booktitle = {XXX},
  year      = {2023},
}
```
-->

![](/figs/Figure1.png)

## Acknowledgment
This implementation is bulit upon [DINO](https://github.com/IDEA-Research/DINO/) and [RemoteSensingTeacher](https://github.com/h751410234/RemoteSensingTeacher).

## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 16.04
* Python: 3.10.9
* CUDA: 11.8
* PyTorch: 2.0.1
* torchvision: 0.15.2

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [DAcoco.py](./datasets/DAcoco.py)

```
'dateset's name': {
    'train_img'  : '',  #train image dir
    'train_anno' : '',  #train coco format json file
    'val_img'    : '',  #val image dir
    'val_anno'   : '',  #val coco format json file
},
```
- Add domain adaptation direction within the script [__init__.py](./datasets/__init__.py). During training, the domain adaptation direction will be automatically parsed and corresponding data will be loaded. 
```
DAOD_dataset = [
XXXX
]
```

## Training / Evaluation / Inference
We provide training script on single node as follows.
- Training with single GPU
```
python main.py --config_file {CONFIG_FILE}
```
- Training with Multi-GPU
```
GPUS_PER_NODE={NUM_GPUS} ./tools/run_dist_launch.sh {NUM_GPUS} python main.py --config_file {CONFIG_FILE}
```

We provide evaluation script to evaluate pre-trained model. 
- Evaluation Model.
```
python evaluation.py --config_file {CONFIG_FILE} --opts EVAL True RESUME {CHECKPOINT_FILE}
```
- Evaluation EMA Model.
```
python evaluation.py --config_file {CONFIG_FILE} --opts EVAL True SSOD.RESUME_EMA {CHECKPOINT_FILE}
```

We provide inference script to visualize detection results. See [inference.py](inference.py) for details
- Inference Model.
```
python inference.py --config_file {CONFIG_FILE} --img_dir {INPUT_IMAGE_DIR} --output_dir {SAVE_RESULT_DIR} --opts RESUME {CHECKPOINT_FILE}
```
- Inference EMA Model.
```
python inference_ema.py --config_file {CONFIG_FILE} --img_dir {INPUT_IMAGE_DIR} --output_dir {SAVE_RESULT_DIR} --opts SSOD.RESUME_EMA {CHECKPOINT_FILE}
```

## Pre-trained models
We provide specific experimental configurations and pre-trained models to facilitate the reproduction of our results. 
You can learn the details of DATR through the paper, and please cite our papers if the code is useful for your papers. Thank you!

Task | mAP50  | Config | Model 
------------| ------------- | -------------| -------------
**Cityscapes to Foggy Cityscapes**  | XXX | [cfg](./config/DA/DINO_4scale_C2F.py) | [model](https://pan.baidu.com/s/1-jg-3vTAo06t7yNM3NU8WQ?pwd=w3x4)
**Sim10k to Cityscapes** | XXX | [cfg](./config/DA/DINO_4scale_sim2cityscapes.py) | [model](https://pan.baidu.com/s/15pdOhVHleLQUMAXiddx9zQ?pwd=gu2z)
**Cityscapes to BDD100K-daytime** | XXXX | [cfg](./config/DA/DINO_4scale_city2BDD100k.py) | [model](https://pan.baidu.com/s/11Z9YGkP0E2mTyT8itKzpfQ?pwd=x4si)

## Result Visualization 



## Reference
https://github.com/IDEA-Research/DINO
https://github.com/h751410234/RemoteSensingTeacher