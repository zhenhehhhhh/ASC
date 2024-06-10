import torch
import torch.nn as nn
from model.transformer.blocks.encoder_layer import EncoderLayer

# 线性编码器
class Encoder(nn.Module):
    def __init__(self, input_d, hidden_d, latent_d):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_d, hidden_d)
        self.fc2 = nn.Linear(hidden_d, hidden_d)
        self.fc3 = nn.Linear(hidden_d, latent_d)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 线性解码器
class Decoder(nn.Module):
    def __init__(self, latent_d, hidden_d, output_d):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_d, hidden_d)
        self.fc2 = nn.Linear(hidden_d, output_d)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 自注意力编码器
class SelfAtteEncoder(nn.Module):
    def __init__(self, head_n, ffn_hidden, layer_n, input_d, drop_prob):
        super(SelfAtteEncoder, self).__init__()
        self.selfattention_layer = nn.ModuleList([EncoderLayer(d_model=input_d,
                                                               ffn_hidden=ffn_hidden,
                                                               n_head=head_n,
                                                               drop_prob=drop_prob)
                                                  for _ in range(layer_n)])
    
    def forward(self, x, src_mask):
        for layer in self.selfattention_layer:
            x = layer(x, src_mask)
        return x

# 自注意力解码器
class SelfAtteDecoder(nn.Module):
    def __init__(self, head_n, ffn_hidden, layer_n, input_d, drop_prob):
        super(SelfAtteDecoder, self).__init__()
        self.selfattention_layer = nn.ModuleList([EncoderLayer(d_model=input_d,
                                                               ffn_hidden=ffn_hidden,
                                                               n_head=head_n,
                                                               drop_prob=drop_prob)
                                                  for _ in range(layer_n)])
    
    def forward(self, x, src_mask):
        for layer in self.selfattention_layer:
            x = layer(x, None)
        return x

# 线性 - 线性 自编码器
class LLAutoEncoder(nn.Module):
    def __init__(self, input_d, hidden_d, latent_d, output_d):
        super(LLAutoEncoder, self).__init__()
        self.encoder = Encoder(input_d, hidden_d, latent_d)
        self.decoder = Decoder(latent_d, hidden_d, output_d)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 自注意力 - 线性 自编码器
class ALAutoEncoder(nn.Module):
    def __init__(self, head_n, ffn_hidden, layer_n, input_d, drop_prob, latent_d, hidden_d, output_d):
        super(ALAutoEncoder, self).__init__()
        self.encoder = SelfAtteEncoder(head_n=head_n, 
                                    ffn_hidden=ffn_hidden, 
                                    layer_n=layer_n, 
                                    input_d=input_d, 
                                    drop_prob=drop_prob)
        self.decoder = Decoder(latent_d, hidden_d, output_d)
    
    def forward(self, x, src_mask):
        x = self.encoder(x, src_mask)
        x = self.decoder(x)
        return x

# 自注意力 - 自注意力 自编码器
class AAAutoEncoder(nn.Module):
    def __init__(self, head_n, ffn_hidden, layer_n, input_d, drop_prob, latent_d, hidden_d, output_d):
        super(ALAutoEncoder, self).__init__()
        self.encoder = SelfAtteEncoder(head_n=head_n, 
                                    ffn_hidden=ffn_hidden, 
                                    layer_n=layer_n, 
                                    input_d=input_d, 
                                    drop_prob=drop_prob)
        self.decoder = SelfAtteDecoder(head_n=head_n, 
                                    ffn_hidden=ffn_hidden, 
                                    layer_n=layer_n, 
                                    input_d=input_d, 
                                    drop_prob=drop_prob)
    
    def forward(self, x, src_mask):
        x = self.encoder(x, src_mask)
        x = self.decoder(x, src_mask)
        return x