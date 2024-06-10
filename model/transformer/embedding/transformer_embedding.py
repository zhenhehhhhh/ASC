from torch import nn
from transformer.embedding.position_encoding import *
from transformer.embedding.token_embedding import *

class TransformerEmbedding(nn.Module):
    # token embedding + positional encoding
    # 位置编码可以为网络提供位置信息
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        # vocab_size: vocabulary 的大小
        # d_model: model 的维度
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)