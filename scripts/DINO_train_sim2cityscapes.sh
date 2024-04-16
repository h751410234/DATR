export CUDA_VISIBLE_DEVICES=0 && python main.py \
	--output_dir logs/DINO_sim10k2city_self_training/R50-MS4_ori_conf1 -c config/nature/DINO_4scale_sim2cityscapes.py  \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0


#export CUDA_VISIBLE_DEVICES=2 && python main.py \
#	--output_dir logs/DINO/S50-MS4 -c config/nature/DINO_4scale_swin.py  \
#	--options dn_scalar=100 embed_init_tgt=TRUE \
#	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
#	dn_box_noise_scale=1.0 backbone_dir=/data2/NCUT/个人文件夹/HJH/北理项目/域适应目标检测/第二篇论文/论文定稿程序
