# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class SelfFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        decoderCFA_layer = DecoderCFALayer(d_model=128, dim_kv=128, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
        decoderCFA_norm = nn.LayerNorm(normalized_shape=128)
        self.id_fusion_layer = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

    def forward(self, id_feature, attr_feature):
        attr_feature = attr_feature.repeat(1, 4)
        ids = torch.split(id_feature, 16, dim=0)
        attrs = torch.split(attr_feature, 16, dim=0)
        outs = []
        for f, a in zip(ids, attrs):
            feature = torch.cat((f, a), dim=1)
            
            feature = feature.permute(1, 0, 2)
            out = self.id_fusion_layer(feature, feature)
            out = out.permute(1, 0, 2)
            
            outs.append(out.clone)
        return torch.cat(outs, dim=0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class FusionModule(nn.Module):
    def __init__(self, feature_dim=128, attr_dim=32, n=1):
        super().__init__()
        fusion_layer = FusionLayer(feature_dim, attr_dim)
        self.layers = _get_clones(fusion_layer, n)

        decoderCFA_layer = DecoderCFALayer(d_model=feature_dim, dim_kv=32, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
        decoderCFA_norm = nn.LayerNorm(normalized_shape=feature_dim)
        self.id_fusion_layer = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

    def forward(self, id_feature, attr_feature):
        # 设置为batch_size最大16
        ids = torch.split(id_feature, 16, dim=0)
        attrs = torch.split(attr_feature, 16, dim=0)
        id_outs = []
        for id_out, attr_out in zip(ids, attrs):
            for layer in self.layers:
                id_out, attr_out = layer(id_out, attr_out)
            
            id_out = id_out.permute(1, 0, 2)
            attr_out = attr_out.permute(1, 0, 2)
            id_out = self.id_fusion_layer(id_out, attr_out)
            id_out = id_out.permute(1, 0, 2)

            id_outs.append(id_out.clone())
        out = torch.cat(id_outs, dim=0)
        return out

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p.unsqueeze(0))

class SingleFusionModule(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        decoderCFA_layer = DecoderCFALayer(d_model=feature_dim, dim_kv=32, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
        decoderCFA_norm = nn.LayerNorm(normalized_shape=feature_dim)
        self.id_fusion_layer = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

    def forward(self, id_feature, attr_feature):
        ids = torch.split(id_feature, 16, dim=0)
        attrs = torch.split(attr_feature, 16, dim=0)
        id_outs = []
        for id_out, attr_out in zip(ids, attrs):
            id_out = id_out.permute(1, 0, 2)
            attr_out = attr_out.permute(1, 0, 2)
            id_out = self.id_fusion_layer(id_out, attr_out)
            id_out = id_out.permute(1, 0, 2)
            id_outs.append(id_out.clone())
        out = torch.cat(id_outs, dim=0)
        return out
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p.unsqueeze(0))


class FusionLayer(nn.Module):
    def __init__(self, feature_dim=128, attr_dim=32):
        super().__init__()
        decoderCFA_layer = DecoderCFALayer(d_model=feature_dim, dim_kv=32, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
        decoderCFA_norm = nn.LayerNorm(normalized_shape=feature_dim)
        self.id_fusion_layer = Decoder(decoderCFA_layer, decoderCFA_norm) 
        
        decoderCFA_layer = DecoderCFALayer(d_model=attr_dim, dim_kv=128, nhead=8, dim_feedforward=256, dropout=0.1, activation='relu')
        decoderCFA_norm = nn.LayerNorm(normalized_shape=attr_dim)
        self.attribute_fusion_layer = Decoder(decoderCFA_layer, decoderCFA_norm)

        self.feature_dim = feature_dim
        self.attribute_dim = attr_dim
    
    def forward(self, id_feature, attr_feature):
        id_input = id_feature.permute(1, 0, 2)
        attr_input = attr_feature.permute(1, 0, 2)

        # avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        # ad_id_feature = avg_pool(id_feature.clone()).permute(1, 0, 2)
        # fd_attr_feature = attr_feature.clone().repeat(1, 1, 4).permute(1, 0, 2)
        id_feature = self.id_fusion_layer(id_input, attr_input)
        attr_feature = self.attribute_fusion_layer(attr_input, id_input)

        id_feature = id_feature.permute(1, 0, 2)
        attr_feature = attr_feature.permute(1, 0, 2)
        return id_feature, attr_feature

# 套一个decoder层，在decoderCFA_layer后面加一个norm函数
class Decoder(nn.Module):
    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

# 输入一个embedding和一个memory，输出一个embedding
class DecoderCFALayer(nn.Module):

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
