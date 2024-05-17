# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test, train_one_epoch_with_self_training
from models.dino import EMA



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='city2bdd100k')
    parser.add_argument('--coco_path', type=str, default='')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.strong_aug:    #是否创建半监督强增广dataset
        dataset_train_strong_aug = build_dataset(image_set='train', args=args,strong_aug = True)
    else:
        dataset_train_strong_aug = None


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if dataset_train_strong_aug is not None:  # 半监督强增广使用
            sampler_train_strong_aug = DistributedSampler(dataset_train_strong_aug)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if dataset_train_strong_aug is not None:  # 半监督强增广使用
            sampler_train_strong_aug = torch.utils.data.RandomSampler(dataset_train_strong_aug)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_da, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if dataset_train_strong_aug is not None:  # 半监督强增广使用
        batch_sampler_train_strong_aug = torch.utils.data.BatchSampler(
        sampler_train_strong_aug, args.batch_size, drop_last=True)

        data_loader_train_strong_aug = DataLoader(dataset_train_strong_aug, batch_sampler=batch_sampler_train_strong_aug,
                                       collate_fn=utils.collate_fn_da, num_workers=args.num_workers)
    else:
        data_loader_train_strong_aug = None


    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['ema_model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return


    #==========self-training准备工作，创建EMA Teacher模型=====================
    #----EMA
    #ema_teacher= ModelEma(model, args.ema_decay_teacher) #teacher model
    ema_teacher= EMA.ModelEMA(model, decay = args.ema_decay_teacher) #teacher model
    best_ema_model = None                                #由于self-training训练存在较大波动，用于保存最优模型
    #========best_ema_model 指标记录=======
    #student model(原版模型) 指标记录
    best_checkpoint_fitness = 0
    #最终的最优模型
    best_ema_model_fitness = 0
    cache_best_ema_model_epoch = 0
    # teacher model 指标记录
    best_ema_teacher_fitness = 0
    cache_best_ema_teacher_epoch = 0

    # ---记录评估指标---
    ema_teacher_eval = []
    best_ema_model_eval = []
    #-----------


    print("Start training")
    args.start_epoch = 36    #----修改初始位置用于DEBUG
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):

        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # ----1.不采用self-training训练
        if epoch < cfg.burn_epochs:
            #when lr drop ，加载最优模型
            if epoch == cfg.lr_drop:
                checkpoint = torch.load(os.path.join(output_dir, 'best_ema_teacher.pth'), map_location='cpu')
                if not args.distributed:
                    model_without_ddp.load_state_dict(checkpoint['ema_model'], strict=True)
                else:  # DDP
                    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['ema_model'].items()}
                    model_without_ddp.load_state_dict(state_dict, strict=True)

                # eval
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )

            #---标准训练 with Domain Adaptation
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)


        # ----2.加入伪标签,self-training训练
        else:
            if args.distributed:
               sampler_train.set_epoch(epoch)
               sampler_train_strong_aug.set_epoch(epoch)

            #----准备工作
            if epoch == cfg.burn_epochs:
                # 考虑重新设置学习率，学习率*10（待实现）
                print('学习率:')
                for p in optimizer.param_groups:
                    print(p['lr'])


                #（0）----------加载最优检测模型----------------
                checkpoint = torch.load(os.path.join(output_dir, 'best_ema_teacher.pth'), map_location='cpu')
                if not args.distributed:
                    model_without_ddp.load_state_dict(checkpoint['ema_model'], strict=True)
                    ema_teacher.ema.load_state_dict(checkpoint['ema_model'], strict=True)
                else:  # DDP
                    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['ema_model'].items()}
                    model_without_ddp.load_state_dict(state_dict, strict=True)
                    ema_teacher.ema.load_state_dict(state_dict, strict=True)

                # 评估，用于debug
                test_stats, coco_evaluator = evaluate(
                    ema_teacher.ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
                     )

                #（1）----------由于学生模型学习可能存在不稳定情况，创建额外的EMA用于保存最优结果---------------)
                best_ema_model = EMA.CosineEMA(ema_teacher.ema,decay_start =  args.ema_decay_best_model,
                                               total_epoch = args.epochs - args.burn_epochs) #使用较大的ema_decay加快模型学习效率

            #（2）----------self-training训练 with Domain Adaptation---------------)
            train_stats = train_one_epoch_with_self_training(
                model, ema_teacher ,criterion, data_loader_train, data_loader_train_strong_aug,optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)

        # ----3.训练后，使用EMA更新teacher model和 best_ema_model
        #（1）---------- 更新teacher model -----------------
        ema_teacher.update(model)
        #（2）---------- 更新best_ema_model model -----------------
        if best_ema_model:
            best_ema_model.update_decay(epoch - args.burn_epochs)  # 更新semi_ema的decay
            best_ema_model.update(ema_teacher.ema)

        # ----4.评估并保存模型
        #（1）---------- 官方保存每一个epoch student model-----------------
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

        #（2）---------- 评估结果-----------------
        #----官方原版student eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        if test_stats['coco_eval_bbox'][1] >= best_checkpoint_fitness:
            best_checkpoint_fitness = test_stats['coco_eval_bbox'][1]
            cache_best_checkpoint_epoch = epoch  # 记录epoch数

        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        #----增添EMA模型评估
        #存在best_ema则评估best_ema
        if epoch >= args.burn_epochs:
            test_stats_best_ema, coco_evaluator_best_ema = evaluate(
                best_ema_model.ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
        else:#不存在则评估ema_teacher
            test_stats_ema_teacher, coco_evaluator_ema_teacher = evaluate(
                ema_teacher.ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )

        #----增添保存最优模型
        if args.output_dir and utils.is_main_process():
            # 1）存储最佳best_ema
            if epoch >= args.burn_epochs:
                best_ema_model_eval.append(test_stats_best_ema['coco_eval_bbox'][1])
                # 记录结果
                with open(output_dir / "best_ema_model_eval.txt", 'w') as f:
                    for i in best_ema_model_eval:
                        f.write('%s\n' % i)

                if test_stats_best_ema['coco_eval_bbox'][1] >= best_ema_model_fitness:
                    best_ema_model_fitness = test_stats_best_ema['coco_eval_bbox'][1]
                    checkpoint_path = output_dir / 'best_ema_model.pth'
                    cache_best_ema_model_epoch = epoch  # 记录epoch数
                    utils.save_on_master({
                        'ema_model': best_ema_model.ema.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

            #2）存储最佳ema teacher
            if epoch < args.burn_epochs:
                ema_teacher_eval.append(test_stats_ema_teacher['coco_eval_bbox'][1])
                #记录结果
                with open(output_dir / "ema_teacher_eval.txt", 'w') as f:
                    for i in ema_teacher_eval:
                        f.write('%s\n'%i)

                if test_stats_ema_teacher['coco_eval_bbox'][1] >= best_ema_teacher_fitness:
                    best_ema_teacher_fitness = test_stats_ema_teacher['coco_eval_bbox'][1]
                    checkpoint_path = output_dir / 'best_ema_teacher.pth'
                    cache_best_ema_teacher_epoch = epoch  # 记录epoch数
                    utils.save_on_master({
                        'ema_model': ema_teacher.ema.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

            # （3）记录日志
            with open(output_dir / "log_best.txt", 'w') as f:
                f.write('best_checkpoint -->  map50:%s , epoch:%s\n' % (
                best_checkpoint_fitness, cache_best_checkpoint_epoch))
                f.write(
                    'best_semi_ema -->  map50:%s , epoch:%s\n' % (best_ema_model_fitness, cache_best_ema_model_epoch))
                f.write('best_teacher -->  map50:%s , epoch:%s\n' % (best_ema_teacher_fitness, cache_best_ema_teacher_epoch))

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
