import numpy as np
import os
import cv2
import torch
import torchvision.transforms as transforms
from util import box_ops
import time
from torchvision.ops.boxes import batched_nms

def _make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_unlabel_img(nestedtensor):
    images,masks = nestedtensor.decompose()
    b,c,h,w = images.shape
    unlabel_b = b // 2
    unlabel_img = images[unlabel_b:,:,:,:]
    return unlabel_img


def get_pseudo_label_via_threshold(results,threshold = 0.8):
    cache_idx_list = []
    cache_labels_dict = {}
    cache_boxes_dict = {}
    cache_scores_dict = {}
    #---取消多尺度训练偶尔出现labels为背景的情况,添加一维用于除去

  #  threshold = np.append(threshold,1.0)


   # print(threshold)
    for n,result in enumerate(results):  #每张图的预测结果
        #{'scores': s, 'labels': l, 'boxes': b}
        # 对应每个类别的 置信度阈值
        threshold_for_class = torch.from_numpy(threshold[result['labels'].cpu().numpy()]).to(result['scores'].device)
        #print(threshold_for_class)
        scores = result['scores']
        vaild_idx = scores >= threshold_for_class
        vaild_labels = result['labels'][vaild_idx]
        vaild_boxes = result['boxes'][vaild_idx]
        vaild_scores = result['scores'][vaild_idx]
        if len(vaild_labels) > 0 :
            cache_idx_list.append(n)
            cache_labels_dict[n] = vaild_labels
            cache_boxes_dict[n] = vaild_boxes
            cache_scores_dict[n] = vaild_scores
    return cache_idx_list,cache_labels_dict,cache_boxes_dict,cache_scores_dict

def deal_pesudo_label(unlabel_target_list,idx_list,pesudo_labels_dict,pesudo_boxes_dict,scores_dcit):
    unlabel_target_format_dict = {}
    for i in idx_list:
        #-----target记录
        cache_unlabel_target_format = {}
        unlabel_target = unlabel_target_list[i]
        cache_unlabel_target_format['labels'] = pesudo_labels_dict[i]
        cache_unlabel_target_format['boxes'] = pesudo_boxes_dict[i]
        cache_unlabel_target_format['scores'] = scores_dcit[i]
        cache_unlabel_target_format['image_id'] = unlabel_target['image_id']
        cache_unlabel_target_format['area'] = unlabel_target['area']
        cache_unlabel_target_format['iscrowd'] = unlabel_target['iscrowd']
        cache_unlabel_target_format['orig_size'] = unlabel_target['orig_size']
        cache_unlabel_target_format['size'] = unlabel_target['size']
        unlabel_target_format_dict[i] = cache_unlabel_target_format
    return unlabel_target_format_dict

def rescale_pseudo_targets(unlabel_samples_img,unlabel_pseudo_targets,nms_th = 0.7):
    _b,_c,_h,_w = unlabel_samples_img.shape

    for k,v in unlabel_pseudo_targets.items():
        _h_real, _w_real = unlabel_pseudo_targets[k]['size'].cpu().numpy()

        unlabel_pseudo_targets[k]['boxes'] = box_ops.box_cxcywh_to_xyxy(unlabel_pseudo_targets[k]['boxes'])

        #（1）恢复为原图坐标
        unlabel_pseudo_targets[k]['boxes'][:,[0,2]] = unlabel_pseudo_targets[k]['boxes'][:,[0,2]] * _w
        unlabel_pseudo_targets[k]['boxes'][:,[1,3]] = unlabel_pseudo_targets[k]['boxes'][:,[1,3]] * _h
        #（2）NMS
        keep_inds = batched_nms(unlabel_pseudo_targets[k]['boxes'],
                                unlabel_pseudo_targets[k]['scores'],
                                unlabel_pseudo_targets[k]['labels'], nms_th)[:100]
        unlabel_pseudo_targets[k]['boxes'] = unlabel_pseudo_targets[k]['boxes'][keep_inds]
        unlabel_pseudo_targets[k]['scores'] = unlabel_pseudo_targets[k]['scores'][keep_inds]
        unlabel_pseudo_targets[k]['labels'] = unlabel_pseudo_targets[k]['labels'][keep_inds]
        #（3）比例缩放
        unlabel_pseudo_targets[k]['boxes'] = box_ops.box_xyxy_to_cxcywh(unlabel_pseudo_targets[k]['boxes'])
        unlabel_pseudo_targets[k]['boxes'][:,[0,2]] = unlabel_pseudo_targets[k]['boxes'][:,[0,2]]  / _w_real
        unlabel_pseudo_targets[k]['boxes'][:,[1,3]] = unlabel_pseudo_targets[k]['boxes'][:,[1,3]]  / _h_real
    return unlabel_pseudo_targets

def spilt_output(output_dict):
    source_dict = {}
    pesudo_dict = {}
    for k,v in output_dict.items():
        if 'target' in k :
            pesudo_dict[k] = v
        else:
            source_dict[k] = v
    return source_dict,pesudo_dict


def get_valid_output(target_outputs,target_pseudo_labels_dict,idx):
    valid_target_outputs = {}
    for k,v in target_outputs.items():
        if 'pred' in k:
            valid_target_outputs[k] = v[idx,:,:]
        elif 'aux_outputs_target' in k:
            cache_list = []
            for sub_v_dict in v:
                cache_dict = {}
                cache_dict['pred_logits'] = sub_v_dict['pred_logits'][idx,:,:]
                cache_dict['pred_boxes'] = sub_v_dict['pred_boxes'][idx,:,:]
                cache_list.append(cache_dict)
            valid_target_outputs[k] = cache_list

        elif 'interm_outputs_target' in k:
            cache_dict = {}
            cache_dict['pred_logits'] = v['pred_logits'][idx,:,:]
            cache_dict['pred_boxes'] = v['pred_boxes'][idx,:,:]
            valid_target_outputs[k] = cache_dict

        elif 'interm_outputs_for_matching_pre_target' in k:
            cache_dict = {}
            cache_dict['pred_logits'] = v['pred_logits'][idx,:,:]
            cache_dict['pred_boxes'] = v['pred_boxes'][idx,:,:]
            valid_target_outputs[k] = cache_dict

        else:
            assert '不存在的输出结果'

    #处理伪标签格式，用于后续损失计算
    target_pseudo_labels = []
    for k,v in target_pseudo_labels_dict.items():
        target_pseudo_labels.append(v)

    return valid_target_outputs,target_pseudo_labels



#======================可视化DEBUG使用=============================

def Denormalize(img):
    # 这是归一化的 mean 和std
    channel_mean = torch.tensor([0.485, 0.456, 0.406])
    channel_std = torch.tensor([0.229, 0.224, 0.225])
    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    # 归一化和反归一化生成器
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    de_img = denormalizer(img)
    return de_img

def draw_img(img,unlabel_samples_img_strong_aug,data_dict,unlabel_target,save_dir):
    _h, _w = data_dict['size'].cpu().numpy()
    boxes = data_dict['boxes'].cpu().numpy()  #中心点+宽高
    boxes[:,[0,2]] *= _w
    boxes[:,[1,3]] *= _h
    img = img.copy()
    img2 = img.copy()
    #----for real labels
    boxes_label = unlabel_target['boxes'].cpu().numpy()  #中心点+宽高
    boxes_label[:,[0,2]] *= _w
    boxes_label[:,[1,3]] *= _h

    for i, box in enumerate(boxes_label):
        cls = unlabel_target['labels'][i].cpu().numpy()
        x_c, y_c, w, h = [int(i) for i in box]
        x1, y1, x2, y2 = x_c - w // 2, y_c - h // 2, x_c + w // 2, y_c + h // 2
        img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, 'label.jpg'), img2)
    #-------------

    if unlabel_samples_img_strong_aug is not None:
        unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug.copy()
        for i,box in enumerate(boxes):
            cls = data_dict['labels'][i].cpu().numpy()
            x_c,y_c,w,h = [int(i) for i in box]
            x1,y1,x2,y2 = x_c - w//2,y_c - h // 2 ,x_c + w//2,y_c + h // 2
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            unlabel_samples_img_strong_aug = cv2.rectangle(unlabel_samples_img_strong_aug,(x1,y1),(x2,y2),(0,0,255),2)
       # cv2.imshow('a',img)
       # cv2.imshow('b',unlabel_samples_img_strong_aug)
        cv2.imwrite(os.path.join(save_dir,'a.jpg'),img)
        cv2.imwrite(os.path.join(save_dir,'b.jpg'),unlabel_samples_img_strong_aug)




        print('停止')
        time.sleep(5000000)



def show_pesudo_label_with_gt(unlabel_img_array,unlabel_pseudo_targets,unlabel_targets,idx_list,unlabel_samples_img_strong_aug_array,save_dir = './show_pseudo'):
    _make_dir(save_dir)

    for n,idx in enumerate(idx_list):
        unlabel_img = unlabel_img_array[idx].detach().cpu()  #根据索引找
        unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug_array[idx].detach().cpu()  #根据索引找
        unlabel_pseudo_target = unlabel_pseudo_targets[idx] #根据索引找
        unlabel_target = unlabel_targets[idx]  #根据索引找

        # 对图像进行反归一化
        unlabel_img = Denormalize(unlabel_img).numpy()
        unlabel_img *= 255.0
        unlabel_img = unlabel_img.transpose(1,2,0).astype(np.uint8)
        if unlabel_samples_img_strong_aug is not None:
            unlabel_samples_img_strong_aug = Denormalize(unlabel_samples_img_strong_aug).numpy()
            unlabel_samples_img_strong_aug *= 255.0
            unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug.transpose(1, 2, 0).astype(np.uint8)

        draw_img(unlabel_img,unlabel_samples_img_strong_aug,unlabel_pseudo_target,unlabel_target,save_dir)
