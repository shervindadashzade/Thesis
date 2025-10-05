import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.register_buffer('mask', torch.tensor(mask))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

class GOAutoEncoder(nn.Module):
    def __init__(self, input_mask, mask_4to3, mask_3to2):
        super().__init__()
        self.encoder = nn.Sequential(
            MaskedLinear(input_mask.shape[1], input_mask.shape[0], input_mask),
            nn.GELU(),
            MaskedLinear(mask_4to3.shape[1], mask_4to3.shape[0], mask_4to3),
            nn.GELU(),
            MaskedLinear(mask_3to2.shape[1], mask_3to2.shape[0], mask_3to2),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            MaskedLinear(mask_3to2.shape[0], mask_3to2.shape[1], mask_3to2.T),
            nn.GELU(),
            MaskedLinear(mask_4to3.shape[0], mask_4to3.shape[1], mask_4to3.T),
            nn.GELU(),
            MaskedLinear(input_mask.shape[0], input_mask.shape[1], input_mask.T)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        return self.encoder(x)
