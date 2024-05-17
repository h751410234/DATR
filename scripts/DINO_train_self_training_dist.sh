export CUDA_VISIBLE_DEVICES=0,1,2,3 && python -m torch.distributed.launch --master_port=29603 --nproc_per_node=4 --use_env main_teacher.py \
--dataset_file sim2city\
--output_dir logs/DINO_Sim10k2Cityscapes_self_training/R50_ms4 -c config/DA/Sim10k2Cityscapes/DINO_4scale_sim2cityscapes_self_training.py \
--options dn_scalar=100 embed_init_tgt=TRUE \
dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
dn_box_noise_scale=1.0


