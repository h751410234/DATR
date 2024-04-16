# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
#---遥感
from .DAcoco import build_xView2DOTA_DA,build_city_DA,build_sim2city_DA,build_voc2clipart1k_DA,build_city2BDD_DA,build_voc2watercolor_DA



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args,strong_aug = False):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)

    #-----增添自然
    if args.dataset_file == 'city':
        return build_city_DA(image_set, args,strong_aug)

    if args.dataset_file == 'sim2city':
        return build_sim2city_DA(image_set, args,strong_aug)

    if args.dataset_file == 'city2bdd100k':
        return build_city2BDD_DA(image_set, args,strong_aug)

    raise ValueError(f'dataset {args.dataset_file} not supported')
