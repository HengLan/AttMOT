from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        # self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.classifier = nn.Linear(32, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.BCELoss = nn.BCELoss()
        self.AttrLoss = F.binary_cross_entropy_with_logits

        self.init_weight()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def init_weight(self):
        # gta-attr-dataset-03的正负样本比例
        rate = [0.4039, 0.0812, 0.5557, 0.3444, 0.2623, 0.4372, 0.2492, 0.2512, 0.1254, 0.2522, 0.01, 0.0634, 0.0178, 0.0259, 0.1863, 0.3012, 
                0.1706, 0.071, 0.0942, 0.0803, 0.1113, 0.0662, 0.0703, 0.0296, 0.3582, 0.1491, 0.0098, 0.0304, 0.017, 0.3129, 0.0268, 0.0807]
        # gta-attr-dataset-04的正负样本比例
        # rate = [0.3978, 0.0749, 0.5687, 0.3386, 0.2656, 0.4244, 0.2575, 0.2558, 0.1269, 0.2501, 0.01, 0.0669, 0.0189, 0.028, 0.1788, 0.3116, 
        #         0.1684, 0.0679, 0.0939, 0.0793, 0.1154, 0.0666, 0.0702, 0.0306, 0.3614, 0.1534, 0.0088, 0.0298, 0.0178, 0.3092, 0.0266, 0.0794]
        weight_pos = []
        weight_neg = []
        for idx, v in enumerate(rate):
            weight_pos.append(math.exp(1.0 - v))
            weight_neg.append(math.exp(v))
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss, attr_loss = 0, 0, 0, 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            # 只进行计算语义属性损失的部分
            
            # if not opt.mse_loss:
            #     output['hm'] = _sigmoid(output['hm'])

            # hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            # if opt.wh_weight > 0:
            #     wh_loss += self.crit_reg(
            #         output['wh'], batch['reg_mask'],
            #         batch['ind'], batch['wh']) / opt.num_stacks

            # if opt.reg_offset and opt.off_weight > 0:
            #     off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
            #                               batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                # id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = output['attr']
                # id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:
                    id_loss += self.IDLoss(id_output, id_target)
            # if opt.attr_weight > 0:
            #     # attr_head = _tranpose_and_gather_feat(output['attr'], batch['ind'])
            #     attr_head = output['attr']
            #     # attr_head = math.sqrt(2) * math.log(32) * F.normalize(attr_head)
            #     # attr_head = attr_head[batch['reg_mask'] > 0].contiguous()
            #     id_attr_target = batch['feature'][batch['reg_mask'] > 0]
            #     # m = nn.Sigmoid()
            #     # attr_output = m(attr_head)
            #     id_attr_target = id_attr_target.float()
            #     # attr_loss += self.BCELoss(attr_output, id_attr_target)

            #     # 设置损失函数权重
            #     weights = torch.zeros(id_attr_target.shape)
            #     for i in range(id_attr_target.shape[0]):
            #         for j in range(id_attr_target.shape[1]):
            #             if id_attr_target.data.cpu()[i, j] == 0:
            #                 weights[i, j] = self.weight_neg[j]
            #             elif id_attr_target.data.cpu()[i, j] == 1:
            #                 weights[i, j] = self.weight_pos[j]
            #             else:
            #                 weights[i, j] = 0 
                
            #     attr_loss += self.AttrLoss(attr_head, id_attr_target, weight=Variable(weights).cuda())
        
        # det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        # if opt.multi_loss == 'uncertainty':
        #     loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * (id_loss + attr_loss) + (self.s_det + self.s_id)
        #     loss *= 0.5
        # else:
        #     loss = det_loss + 0.1 * (id_loss + attr_loss)

        # loss = id_loss + attr_loss
        # loss_stats = {'loss': loss, 'id_loss': id_loss, 'attr_loss': attr_loss}

        # loss = attr_loss * 32
        # loss_stats = {'loss': loss, 'attr_loss': attr_loss}

        loss = id_loss
        loss_stats = {'loss': loss}
        
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        # loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'attr_loss']
        loss_states = ['loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
