import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, feat_dim=128, attr_dim=32):
        super().__init__()
        self.feat_mlp = Adapter(128, 128, 512)
        self.attr_mlp = Adapter(32, 32, 128)
        self.fusion_layer = CrossAttention(d_model=feat_dim, dim_kv=attr_dim, nhead=8)

    def forward(self, feat, attr):
        # print('before:', feat.shape)
        feat = self.feat_mlp(feat)
        attr = self.attr_mlp(attr)
        # print('after mlp:', feat.shape)
        feat = feat.permute(1, 0, 2)
        attr = attr.permute(1, 0, 2)
        # print('after permute:', feat.shape)
        feat = self.fusion_layer(feat, attr)
        feat = feat.permute(1, 0, 2)
        # print('after attention:', feat.shape)
        return feat

class Adapter(nn.Module):
    def __init__(self, num_in, num_out, num_hidden):
        super().__init__()
        self.linear_1 = torch.nn.Linear(num_in, num_hidden)
        self.linear_2 = torch.nn.Linear(num_hidden, num_hidden)
        self.linear_3 = torch.nn.Linear(num_hidden, num_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_kv, dropout=0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=dim_kv, vdim=dim_kv)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
