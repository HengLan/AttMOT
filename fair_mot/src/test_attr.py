import time
import torch
import json
import numpy as np
from sklearn import metrics
from scipy import interpolate
from lib.datasets.dataset.jde import JointDataset
from lib.models.utils import _tranpose_and_gather_feat
from lib.opts import opts
from lib.models.model import create_model, load_model
from torchvision.transforms import transforms as T


def test_attr(opt, batch_size, img_size=(1088, 608), print_interval=100):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1
    test_paths = data_cfg_dict['test_emb']
    dataset_root = data_cfg_dict['root']
    opt.device = torch.device('cuda')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()

    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(opt, dataset_root, test_paths, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=False)
    attributes, id_labels = [], []
    print('Extracting pedestrian attrs...')
    for batch_i, batch in enumerate(dataloader):
        t = time.time()
        output = model(batch['input'].cuda())[-1]
        attr_head = _tranpose_and_gather_feat(output['attr'], batch['ind'].cuda())
        attr_head = attr_head[batch['reg_mask'].cuda() > 0].contiguous()
        m = torch.nn.Sigmoid()
        attr_head = m(attr_head)
        id_target = batch['ids'].cuda()[batch['reg_mask'].cuda() > 0]

        for i in range(0, attr_head.shape[0]):
            if len(attr_head.shape) == 0:
                continue
            else:
                attr, label = attr_head[i], id_target[i].long()
            if label != -1:
                attributes.append(attr)
                id_labels.append(label)
        
        if batch_i % print_interval == 0:
            print(
                'Extracting {}/{}, # of instances {}, time {:.2f} sec.'.format(batch_i, len(dataloader), len(id_labels),
                                                                               time.time() - t))
    print('Computing pairwise similarity...')
    if len(attributes) < 1:
        return None
    attributes = torch.stack(attributes, dim=0).cuda()
    id_labels = torch.LongTensor(id_labels)
    n = len(id_labels)
    print(n, len(attributes))
    assert len(attributes) == n

    pdist = torch.mm(attributes, attributes.t()).cpu().numpy()
    gt = id_labels.expand(n, n).eq(id_labels.expand(n, n).t()).numpy()

    up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
    pdist = pdist[up_triangle]
    gt = gt[up_triangle]

    far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    far, tar, threshold = metrics.roc_curve(gt, pdist)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]
    for f, fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
    return tar_at_far


if __name__ == '__main__':
    opt = opts().init()
    with torch.no_grad():
        tpr = test_attr(opt, batch_size=4)
