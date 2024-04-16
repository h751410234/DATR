# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process
from .DA_utils import decompose_features,grad_reverse,FCDiscriminator_img,get_prototype_class_wise
import time



class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # # for DA trainging
        if self.training:
            #空域
            self.D_img = FCDiscriminator_img(256) # Need to know the channel
            #原型
            self.global_proto = torch.zeros(self.num_classes, 256).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.num_classes).cuda(non_blocking=True)
            self.Proto_D = MLP(hidden_dim, hidden_dim, 1, 3)

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    #def forward(self, samples: NestedTensor, targets:List=None, self_training_flag = False,global_proto = None , global_amount = None):
    def forward(self, samples: NestedTensor, targets:List=None, self_training_flag = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)


        #----训练时, 拆分源域和目标域特征图
        if self.training:
            srcs,masks,poss,srcs_all ,masks_all,poss_all,srcs_target,masks_target,poss_target = decompose_features(srcs,masks,poss)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None


        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss,input_query_label,attn_mask)

        # In case num object=0
        hs[0] += self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])

        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta,self.aux_loss,self._set_aux_loss)



        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)


        # for encoder output

        if hs_enc is not None:
            # prepare intermediate outputs

            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])

            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)
                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        #----DA Process
        if self.training:
            da_output = {}
            #First.==============backbone 特征对齐======================
            D_img_out = []
            for src in srcs_all:  #不同层级
                src = grad_reverse(src)  #GRL
                D_img_out_sub = self.D_img(src)  #Dis
                D_img_out.append(D_img_out_sub)
            da_output['backbone_DA'] = torch.cat([out.flatten(2).transpose(1, 2) for out in D_img_out],dim=1)
            #======================================================

            #Second.==============objcet query Prototypical Contrast Adaptation======================
            #---- For Source Domain
            # 1.--------提取原型特征-----------------
            # (1)得到最后一层object query 结果
            output_source = hs[-1][:,dn_meta['pad_size']:,:]   #source_query_feature :[bs,N,256]
            # (2)得到预测分类结果
            outputs_class_source = out['pred_logits']
            # (3)提取各类别prototype特征
            class_prototypes_source,vaild_class_map_source,global_proto,global_amount,query_mask_source = get_prototype_class_wise(output_source,outputs_class_source,self.num_classes,
                                                                                      global_proto = self.global_proto.detach(),global_amount = self.Amount)
            # (4)更新全局prototype
            self.global_proto = global_proto
            self.Amount = global_amount



            #---- For Target Domain
            # 1.--------对目标域数据进行推理-----------------
            hs_target, reference_target, hs_enc_target, ref_enc_target, init_box_proposal_target = self.transformer(
                srcs_target, masks_target, None, poss_target,
                None, None)
            # 2.--------提取原型特征-----------------
            # (1)得到最后一层object query 结果
            outputs_target = hs_target[-1]
            # (2)得到预测分类结果
            outputs_class_target = self.class_embed[-1](outputs_target)
            # (3)提取各类别prototype特征
            class_prototypes_target,vaild_class_map_target,global_proto,global_amount,query_mask_target = get_prototype_class_wise(outputs_target,outputs_class_target,self.num_classes,
                                                                                      global_proto = self.global_proto.detach(),global_amount = self.Amount)
            # (4)更新全局prototype
            self.global_proto = global_proto
            self.Amount = global_amount


            class_prototypes = torch.cat([class_prototypes_source,class_prototypes_target],dim=0)    #[14,1024]
            class_prototypes_da = self.Proto_D(grad_reverse(class_prototypes)) #[14,1]

            # 3.--------记录 prototype 原型结果-----------------
            da_output['proto_DA'] = {'da_protos': class_prototypes_da,  #concat的原型
                                    'class_map_source': vaild_class_map_source,  #用于记录源域图像中的存在类别
                                    'class_map_target': vaild_class_map_target,  ##用于记录目标域图像中的存在类别
                                     }

            da_output['global_proto_DA'] = {'output_source': class_prototypes_source,
                                            'outputs_target': class_prototypes_target,
                                            'query_mask_source': vaild_class_map_source,
                                            'query_mask_target': vaild_class_map_target,
                                            'global_proto': self.global_proto,
                                           }



            #保存所有跟域适应有关的结果
            out['da_output'] = da_output
#        ---------



            #Third.==============自训练： 目标域结果处理+保存 ============================
            if self_training_flag:
                # In case num object=0
                hs_target[0] += self.label_enc.weight[0, 0] * 0.0

                # deformable-detr-like anchor update
                # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
                outputs_coord_list_target = []
                for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                        zip(reference_target[:-1], self.bbox_embed, hs_target)):
                    layer_delta_unsig = layer_bbox_embed(layer_hs)
                    layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                    layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                    outputs_coord_list_target.append(layer_outputs_unsig)
                outputs_coord_list_target = torch.stack(outputs_coord_list_target)

                outputs_class_target_layers = torch.stack([layer_cls_embed(layer_hs) for
                                             layer_cls_embed, layer_hs in zip(self.class_embed, hs_target)])


                #----增添目标域预测结果，用于计算监督损失
                out['pred_logits_target'] = outputs_class_target_layers[-1]
                out['pred_boxes_target'] = outputs_coord_list_target[-1]

                if self.aux_loss:
                    out['aux_outputs_target'] = self._set_aux_loss(outputs_class_target_layers, outputs_coord_list_target)

                # for encoder output
                if hs_enc_target is not None:
                    # prepare intermediate outputs

                    interm_coord_target = ref_enc_target[-1]
                    interm_class_target = self.transformer.enc_out_class_embed(hs_enc_target[-1])

                    out['interm_outputs_target'] = {'pred_logits': interm_class_target, 'pred_boxes': interm_coord_target}
                    out['interm_outputs_for_matching_pre_target'] = {'pred_logits': interm_class_target,
                                                              'pred_boxes': init_box_proposal_target}

                    # prepare enc outputs
                    if hs_enc_target.shape[0] > 1:
                        enc_outputs_coord_target = []
                        enc_outputs_class_target = []
                        for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(
                                zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc_target[:-1], ref_enc_target[:-1])):
                            layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                            layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                            layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                            layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                            enc_outputs_coord_target.append(layer_enc_outputs_coord)
                            enc_outputs_class_target.append(layer_enc_outputs_class)

                        out['enc_outputs_target'] = [
                            {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class_target, enc_outputs_coord_target)
                        ]
        return out#,global_proto,global_amount

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    #==============增添域适应 Loss
    def loss_da(self, outputs): #----backbone
        B = outputs.shape[0]
        assert B % 2 == 0

        #Source
        outputs_source = outputs[:B //2,:,:]
        loss_outputs_source = F.binary_cross_entropy_with_logits(outputs_source, torch.FloatTensor(outputs_source.data.size()).fill_(
            0).to(outputs_source.device))
        #Target
        outputs_target = outputs[B //2:,:,:]
        loss_outputs_target = F.binary_cross_entropy_with_logits(outputs_target, torch.FloatTensor(outputs_target.data.size()).fill_(
            1).to(outputs_source.device))

        loss = loss_outputs_source + loss_outputs_target

        return loss#.mean()

    #==============增添域适应 Loss
    def loss_proto_da(self, outputs): #----proto

        output_protos = outputs['da_protos']
        assert output_protos.shape[0] % 2 == 0
        class_map_source = outputs['class_map_source']  #[9,1]
        class_map_target = outputs['class_map_target']  #[9,1]
        class_num = class_map_source.shape[0]
        targets = torch.empty_like(output_protos)
        targets[:class_num] = 0
        targets[class_num:] = 1
        #计算损失
        loss = F.binary_cross_entropy_with_logits(output_protos, targets, reduction='none') #[18]
        #得到class_map
        class_map = torch.cat([class_map_source,class_map_target],dim=0).unsqueeze(1)
        loss = loss * class_map
        return loss.mean()


    # ----全局原型损失-----
    def loss_contrast_da(self, outputs):

        # """ -----------验证结果，考虑调研对比学习损失
        # Args:
        # da_output['global_proto_DA'] = {'output_source': output_source,
        #                                 'outputs_target': outputs_target,
        #                                 'query_mask_source': query_mask_source,
        #                                 'query_mask_target': query_mask_target,
        #                                 'global_proto': global_proto,
        #                                 }
        # Returns:
        #
        # """

        #---object query
        query_source = outputs['output_source'] #[class_num,256]
        query_target = outputs['outputs_target'] #[class_num,256]

        #---对应query 掩模图
        query_mask_source = outputs['query_mask_source'] #[class_num]
        query_mask_target = outputs['query_mask_target'] #[class_num]
        #---global_proto
        global_proto = outputs['global_proto']           #[class_num,256]

        #---判断反向传播
        assert not global_proto.requires_grad
        assert not query_mask_source.requires_grad
        assert not query_mask_target.requires_grad
        assert query_source.requires_grad
        assert query_target.requires_grad

        # (0)获取维度
        class_num,C = query_source.shape
        # (1)归一化
        query_source =  F.normalize(query_source, dim=1)
        query_target =  F.normalize(query_target, dim=1)
        global_proto =  F.normalize(global_proto, dim=1)

        # (2)计算相似性
        logits_source = query_source.mm(global_proto.permute(1, 0).contiguous())  # (class_num,C) * (C,class_num) = （class_num,class_num）
        logits_target = query_target.mm(global_proto.permute(1, 0).contiguous())  # (class_num,C) * (C,class_num) = （class_num,class_num）

        # (3)制备标签
        source_label = torch.eye(class_num).to(logits_source.device)
        source_label = source_label * query_mask_source

        target_label = torch.eye(class_num).to(logits_target.device)
        target_label = target_label * query_mask_target

        # (4)计算损失
        ce_criterion = nn.CrossEntropyLoss()
        loss_source = ce_criterion(logits_source, source_label)
        loss_target = ce_criterion(logits_target, target_label)
        loss = loss_source + loss_target

        return loss


    def forward(self, outputs, targets, return_indices=False,target_domain_flag = False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        """
            source_outputs.keys:['pred_logits', 'pred_boxes', 'aux_outputs', 'interm_outputs', 'interm_outputs_for_matching_pre', 'dn_meta', 'da_output']
            target_outputs.keys:['pred_logits_target', 'pred_boxes_target', 'aux_outputs_target', 'interm_outputs_target', 'interm_outputs_for_matching_pre_target']
        """

        #--------调整target domain 结果命名，是其保持与source一致
        if target_domain_flag:
            outputs_without_aux = {k.replace('_target',''): v for k, v in outputs.items() if k != 'aux_outputs_target'}
            #改变pseudo_outputs命名，与监督训练命名规则保持一致
            outputs.update({'pred_boxes': outputs.pop('pred_boxes_target')})
            outputs.update({'pred_logits': outputs.pop('pred_logits_target')})
            device = outputs['pred_logits'].device
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' }
            device = next(iter(outputs.values())).device

        #--------使用匈牙利匹配得到对应索引
        if len(targets) > 0: #存在伪标签
            indices = self.matcher(outputs_without_aux, targets)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)

            if return_indices:
                indices0_copy = indices
                indices_list = []


        else:   #设置虚假值，以保证DDP同步
            indices = None
            num_boxes = torch.as_tensor([1], dtype=torch.float, device=outputs['pred_logits'].device)

        #------不存在伪标签时,导致DDP卡住,因此需要设置indices以保证同步----------
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        if indices is None:
            num_boxes = num_boxes - 1
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if indices is None: #不存在为标签
            return {}
        #---------------------------

        # Compute all the requested losses
        losses = {}

        #---仅源域 计算dn loss ： prepare for dn loss
        if not target_domain_flag:
            dn_meta = outputs['dn_meta']

            if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)
                dn_pos_idx = []
                dn_neg_idx = []
                for i in range(len(targets)):
                    if len(targets[i]['labels']) > 0:
                        t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                        t = t.unsqueeze(0).repeat(scalar, 1)
                        tgt_idx = t.flatten()
                        output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                        output_idx = output_idx.flatten()
                    else:
                        output_idx = tgt_idx = torch.tensor([]).long().cuda()

                    dn_pos_idx.append((output_idx, tgt_idx))
                    dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

                output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
                l_dict = {}
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

                l_dict = {k + f'_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                losses.update(l_dict)

            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs or 'aux_outputs_target' in outputs:
            if target_domain_flag: #针对目标域
                key_aux = 'aux_outputs_target'
            else:                  #针对源域
                key_aux = 'aux_outputs'

            for idx, aux_outputs in enumerate(outputs[key_aux]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # ---仅源域 计算dn loss
                if not target_domain_flag:
                    if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                        aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                        l_dict={}
                        for loss in self.losses:
                            kwargs = {}
                            if 'labels' in loss:
                                kwargs = {'log': False}

                            l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                     **kwargs))

                        l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    else:
                        l_dict = dict()
                        l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs or 'interm_outputs_target' in outputs:
            if target_domain_flag: #针对目标域
                key_aux = 'interm_outputs_target'
            else:                  #针对源域
                key_aux = 'interm_outputs'
            interm_outputs = outputs[key_aux]
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs or 'enc_outputs_target' in outputs:
            if target_domain_flag: #针对目标域
                key_aux = 'enc_outputs_target'
            else:                  #针对源域
                key_aux = 'enc_outputs'
            for i, enc_outputs in enumerate(outputs[key_aux]):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        #===============DA LOSS================
        if 'da_output' in outputs:
            da_output = outputs['da_output']
            #1.backbone_DA
            losses['loss_backbone_DA'] = self.loss_da(da_output['backbone_DA'])

            #2.Proto_DA
            losses['loss_proto_DA'] = self.loss_proto_da(da_output['proto_DA'])

            #3.global contrast_DA
            losses['loss_global_proto_DA'] = self.loss_contrast_da(da_output['global_proto_DA'])
        #======================================
        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]



        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DA training:
    weight_dict['loss_backbone_DA'] = args.da_backbone_loss_coef
    weight_dict['loss_proto_DA'] = args.da_proto_loss_coef
    weight_dict['loss_global_proto_DA'] = args.da_global_proto_coef


    print('************')
    print('loss_backbone_DA:',weight_dict['loss_backbone_DA'])
    print('loss_proto_DA:',weight_dict['loss_proto_DA'])
    print('loss_global_proto_DA:',weight_dict['loss_global_proto_DA'])

    # for self-training:
    weight_dict['loss_self_training'] = args.self_training_loss_coef
    print('loss_self_training:',weight_dict['loss_self_training'])
    print('!!!!!!!!!!!')

    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef


    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
