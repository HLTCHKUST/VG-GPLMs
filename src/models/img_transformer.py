import math
import copy
from typing import Optional, List
import torch
from torch import nn

class ImageTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dim_feedforward=2048):
        super(ImageTransformerEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder = _TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None):
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * l + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)
        else:
            mask = None

        inputs = inputs.permute(1, 0, 2)

        inputs = inputs * math.sqrt(self.d_model)
        inputs = self.pos_encoder(inputs)

        outputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        return [o.permute(1, 0, 2) for o in outputs]


def padTensor(t: torch.Tensor, targetLen: int) -> torch.Tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = [src]

        for mod in self.layers:
            output = mod(outputs[-1], src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            outputs[-1] = self.norm(outputs[-1])

        return outputs[1:]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
