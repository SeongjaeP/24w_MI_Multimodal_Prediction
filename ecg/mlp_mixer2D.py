import torch
import torch.nn as nn
import math
import numpy as np
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.token_mixing = FeedForward(num_patches, token_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.channel_mixing = FeedForward(dim, channel_dim, dropout)


    def forward(self, x):
        # Token mixing
        y = self.layer_norm1(x).transpose(1, 2)
        y = self.token_mixing(y).transpose(1, 2)
        x = x + y 

        y = self.layer_norm2(x)
        y = self.channel_mixing(y)
        x = x + y 

        return x


class MLPMixer2D(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.pe = nn.Parameter(self.pe2d(num_patches, dim), requires_grad=False)

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, num_patches, token_dim, channel_dim, dropout=0.1))

        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))


    def pe2d(self, num_patches, dim):
        pe = torch.zeros(num_patches, dim)
        position = torch.arange(0, num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)


    def forward(self, x):
        x = self.to_patch_embedding(x)
        pe_device = self.pe.to(x.device)
        # PE 적용
        x = x + pe_device

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)

        return self.mlp_head(x)



if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = MLPMixer2D(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]


