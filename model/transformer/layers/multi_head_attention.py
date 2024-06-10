import torch
import torch.nn as nn
from model.transformer.layers.scale_dot_product_attention import *

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        #1 与权重矩阵相乘
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        #2 通过注意力头的数目分裂向量
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        #3 做缩放点积计算相似性
        out, attention = self.attention(q, k, v, mask=mask)
        
        #4 做连接并过线性层
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out
    
    def split(self, tensor):
        # 通过注意力头的数量分割张量
        # tensor [batch_size, length, d_model]
        # return [batch_size, head, length, d_tensor]
        batch_size, length, d_model = tensor.size()
        
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        
        return tensor

    def concat(self, tensor):
        # self.split 的相反函数操作
        # tensor [batch_size, head, length, d_tensor]
        # return [batch_size, length, d_model]
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor