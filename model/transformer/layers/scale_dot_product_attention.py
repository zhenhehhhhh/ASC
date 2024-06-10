import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    # 计算点击注意力
    # query : 给定的我们需要注意的句子
    # key   : 每一个需要与 query 检查关系的句子
    # value : 每一个与 key 相似的句子
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None, e = 1e-12):
        # 输入为 4 维张量 [batch_size, head, length, d_tentor]
        batch_size, head, length, d_tentor = k.size()
        
        #1 计算 q 与 k^T 的点积 -> 相似性
        k_t = k.transpose(2, 3) # 转置
        score = (q @ k_t) / math.sqrt(d_tentor) # 缩放点积
        
        #2 应用掩码
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        
        #3 将他们过一层 softmax 到 [0, 1] 范围
        score = self.softmax(score)
        
        #4 与 v 相乘
        v = score @ v
        
        return v, score