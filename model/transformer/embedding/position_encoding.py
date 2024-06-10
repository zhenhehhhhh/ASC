import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    # 计算正弦编码
    def __init__(self, d_model, max_len, device):
        # d_model : 模型的维度
        # max_len : 最大序列长度
        # device  : 适用的硬件设备
        super(PositionalEncoding, self).__init__()
        
        # 与输入矩阵大小一致
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 无需计算梯度
        
        pos = torch.arrange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # 在一维插入大小为 1 的新维度
        
        # i 代表 d_model 的下标 （例 嵌入层大小为50，则 i = [0, 50]）
        # step = 2 意味着 i * 2
        _2i = torch.arrange(0, d_model, step=2, device=device).float()
        
        # 考虑位置信息计算位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        return self.encoding[:seq_len, :]
        