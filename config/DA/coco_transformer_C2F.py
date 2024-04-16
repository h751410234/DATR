
#针对Cityscapes数据集，防止图像被过度缩小。
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_scales = [int(i*1.5) for i in data_aug_scales]
data_aug_max_size = 2048
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_resize = [int(i*1.5) for i in data_aug_scales2_resize]
data_aug_scales2_crop = [384, 600]
data_aug_scales2_crop = [int(i*1.5) for i in data_aug_scales2_crop]

data_aug_scale_overlap = None

