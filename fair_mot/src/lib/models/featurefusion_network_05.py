import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, attr_dim=32):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.v = nn.Parameter(torch.Tensor(attr_dim, attr_dim))
        self.b = nn.Parameter(torch.Tensor(attr_dim, 1))
        self.init_weight()
    
    def init_weight(self):
        nn.init.kaiming_normal_(self.v, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.b, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, feat, attr):
        feat_ori, attr_ori = feat.clone(), attr.clone()
        feat_out, attr_out = [], [] 
        
        feats = torch.split(feat.squeeze(1), 1, dim=0)
        attrs = torch.split(attr.squeeze(1), 1, dim=0)
        num = len(attrs)
        for i in range(num):
            a = attrs[i]
            # print('a1:', a.shape)
            c = self.sigmoid(torch.mm(self.v, a.t()) + self.b)
            # print('c:', c.shape)
            a = torch.mul(c.t(), a)
            # print('a2:', a.shape)
            attr_out.append(a.clone())
            f = torch.cat((feats[i], a), dim=1)
            # print('f:', f.shape)
            feat_out.append(f.clone())

        feat_out, attr_out = torch.cat(feat_out, dim=0), torch.cat(attr_out, dim=0)
        feat_out, attr_out = feat_out.unsqueeze(1), attr_out.unsqueeze(1)
        # print('feat_out:', feat_out.shape)
        # print('feat_ori:', feat_ori.shape)
        return feat_ori, attr_ori, feat_out, attr_out
