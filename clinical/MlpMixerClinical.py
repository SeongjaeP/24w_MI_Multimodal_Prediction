import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x


class MixerBlock(nn.Module):
    def __init__(self, num_features, num_patches, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(num_features)
        self.mlp1 = MlpBlock(num_patches, tokens_mlp_dim)
        self.layer_norm2 = nn.LayerNorm(num_features)
        self.mlp2 = MlpBlock(num_features, channels_mlp_dim)

    def forward(self, x):
        y = self.layer_norm1(x).transpose(1, 2)
        y = self.mlp1(y).transpose(1, 2)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp2(y)
        x = x + y

        return x


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_patches, num_features, num_blocks, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixer, self).__init__()
        self.num_patches = num_patches
        self.num_features = num_features
        self.mixer_blocks = nn.Sequential(*[MixerBlock(num_features, num_patches, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(num_features)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.view(-1, self.num_patches, self.num_features)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        
        return x