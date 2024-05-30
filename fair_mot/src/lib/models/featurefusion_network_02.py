import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class FusionModule(nn.Module):
    def __init__(self, feat_dim=128, attr_dim=32):
        super().__init__()
        self.feat_encoder = Encoder(d_model=feat_dim, nhead=8, dropout=0.1)
        self.attr_encoder = Encoder(d_model=attr_dim, nhead=8,  dropout=0.1)
        self.feat_decoder = Decoder(d_model=feat_dim, dim_kv=attr_dim, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
        # self.attr_decoder = Decoder(d_model=attr_dim, dim_kv=feat_dim, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
        self.feat_norm = nn.LayerNorm(normalized_shape=feat_dim)
        # self.attr_norm = nn.LayerNorm(normalized_shape=attr_dim)
    
    # def forward(self, feat, attr):
    #     feats = torch.split(feat, 16, dim=0)
    #     attrs = torch.split(attr, 16, dim=0)
    #     feat_outs, attr_outs = [], []
    #     for f, a in zip(feats, attrs):
    #         f = f.permute(1, 0, 2)
    #         a = a.permute(1, 0, 2)

    #         f = self.feat_encoder(f)
    #         a = self.attr_encoder(a)
    #         f = f + self.feat_decoder(f, a)
    #         # a = a + self.attr_decoder(a, f)
    #         f = self.feat_norm(f)
    #         # a = self.attr_norm(a)

    #         f = f.permute(1, 0, 2)
    #         # a = a.permute(1, 0, 2)
    #         feat_outs.append(f.clone())
    #         # attr_outs.append(a.clone())
    #     # return torch.cat(feat_outs, dim=0), torch.cat(attr_outs, dim=0)
    #     return torch.cat(feat_outs, dim=0)

    def forward(self, feat, attr):
        feat = feat.permute(1, 0, 2)
        attr = attr.permute(1, 0, 2)

        feat = self.feat_encoder(feat)
        attr = self.attr_encoder(attr)
        feat = feat + self.feat_decoder(feat, attr)
        feat = self.feat_norm(feat)

        feat = feat.permute(1, 0, 2)
        return feat

    
class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos_src: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos_src)
        tmp = self.self_attn(q, k, value=src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(tmp)
        src = self.norm(src)
        return src

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_kv, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=dim_kv, vdim=dim_kv)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
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
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
