import torch
import torch.nn as nn

from model.transformer.layers.layer_norm import *
from model.transformer.layers.multi_head_attention import *
from model.transformer.layers.position_wise_feed_forward import *
from model.transformer.layers.scale_dot_product_attention import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x, src_mask):
        #1 计算自注意力
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        #2 add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        #3 位置前馈网络
        _x = x
        x = self.ffn(x)
        
        #4 add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x