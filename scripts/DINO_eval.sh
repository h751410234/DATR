python main.py \
  --dataset_file sim2city\
  --output_dir logs/TEST/R50-MS4-eval \
	-c config/DA/Sim10k2Cityscapes/DINO_4scale_sim2cityscapes.py \
	--eval --resume model.pth \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
