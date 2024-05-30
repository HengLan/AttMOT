from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2
import math

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
import torch.nn.functional as F
from lib.models.model import create_model, load_model
from lib.datasets.dataset.jde import JointDataset, collate_fn
from lib.models.utils import _tranpose_and_gather_feat
from lib.utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from lib.opts import opts
from lib.models.decode import mot_decode
from lib.utils.post_process import ctdet_post_process


def test_attr(
        opt,
        batch_size=16,
        img_size=(1088, 608),
        print_interval=40,
):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1
    test_paths = data_cfg_dict['test_emb']
    dataset_root = data_cfg_dict['root']
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(opt, dataset_root, test_paths, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=False)
    embedding, id_labels = [], []
    print('Extracting pedestrain attributes...')
    for batch_i, batch in enumerate(dataloader):
        output = model(batch['input'].cuda())[-1]
        attr_head = _tranpose_and_gather_feat(output['attr'], batch['ind'])
        attr_head = attr_head[batch['reg_mask'].cuda() > 0].contiguous()
        m = torch.nn.Sigmoid()
        attr_head = m(attr_head)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    with torch.no_grad():
        tpr = test_attr(opt, batch_size=4)
