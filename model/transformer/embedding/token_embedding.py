from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        # vocab_size: size of vocabulary
        # d_model: dimensions of model
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)