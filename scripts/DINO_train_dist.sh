#export CUDA_VISIBLE_DEVICES=1,2,3 && python -m torch.distributed.launch --master_port=29603 --nproc_per_node=3 --use_env main.py \
#	--output_dir logs/DINO_city/R50-MS5_conf0.1 -c config/nature/DINO_5scale_city.py \
#	--options dn_scalar=100 embed_init_tgt=TRUE \
#	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
#	dn_box_noise_scale=1.0

export CUDA_VISIBLE_DEVICES=3 && python main.py \
--output_dir logs/DINO_city/R50_ms4_ori -c config/nature/DINO_4scale.py \
--options dn_scalar=100 embed_init_tgt=TRUE \
dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
dn_box_noise_scale=1.0
