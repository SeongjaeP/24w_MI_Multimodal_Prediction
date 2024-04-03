import torch
import torch.nn as nn


# Redefine the FeedForward module to ensure the input dimension matches the first Linear layer's expectation
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # Ensure the input dimension 'dim' matches this layer's expectation
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.net(x)


# No changes needed for ECG
# MixerBlock as it properly configures FeedForward
class ECGMixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):

        super().__init__()

        # Token mixing
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, token_dim, dropout)  # Ensure the first Linear layer of FeedForward matches 'dim'
        )

        # Channel mixing
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )


    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixerForECG(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels, depth, token_dim, channel_dim):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.dim = token_dim * num_channels

        # 패치 임베딩 대신 1D 데이터를 직접 처리하는 레이어로 시작
        self.to_patch_embedding = nn.Linear(num_channels, self.dim)

        # Mixer 블록들
        self.mixer_blocks = nn.ModuleList([
            ECGMixerBlock(self.dim, seq_len, token_dim, channel_dim)
            for _ in range(depth)
        ])

        # 레이어 정규화와 분류를 위한 헤드
        self.layer_norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, num_classes)

    def forward(self, x):
        # [batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.transpose(1, 2)

        # 패치 임베딩
        x = self.to_patch_embedding(x)

        # Mixer 블록 적용
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        # 레이어 정규화
        x = self.layer_norm(x)

        # Global average pooling과 헤드를 통한 분류
        x = x.mean(dim=1)
        x = self.head(x)
        
        return x


if __name__ == "__main__":

    ecg_signal = torch.randn([1, 1, 500])  # Example ECG signal batch with size [batch, channels, length]

    # 'channels'를 'num_channels'로 변경
    model = MLPMixerForECG(num_classes=2, seq_len=500, num_channels=1, depth=6, token_dim=64, channel_dim=256)

    output = model(ecg_signal)

    print(output.shape)  # Expected output shape: [1, num_classes]


