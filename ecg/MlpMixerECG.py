
import torch
import torch.nn as nn


class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x



class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim)
        self.mlp_channel = MlpBlock(hidden_dim, channel_mlp_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)
        y = y.permute(0, 2, 1)  #행렬 전치 수행
        y = self.mlp_token(y)
        y = y.permute(0, 2, 1)
        # Residual connection for token mixing
        x = x + y

        y = self.layer_norm_2(x)
        # Channel mixing across the hidden dimension
        y = self.mlp_channel(y)
        # Residual connection for channel mixing
        x = x + y

        return x


class MlpMixer(nn.Module):
    def __init__(self, num_channels, hidden_dim, mlp_token_dim, mlp_channel_dim, seq_len, num_blocks, num_classes):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks

        # 1D 컨볼루션 임베딩 - 시계열 데이터의 각 포인트를 하나의 '패치'로 처리
        self.conv_embedding = nn.Conv1d(num_channels, hidden_dim, kernel_size=1)

        # PE 추가: 학습 가능한 위치 벡터
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))

        # Mixer 블록
        self.blocks = nn.ModuleList([
            MixerBlock(hidden_dim, seq_len, mlp_token_dim, mlp_channel_dim)
            for _ in range(num_blocks)
        ])

        # 분류기 헤드
        self.head_layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # [batch, channels, seq_len] -> [batch, hidden_dim, seq_len]
        x = self.conv_embedding(x)
        
        # [batch, hidden_dim, seq_len] -> [batch, seq_len, hidden_dim]
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        x = self.head_layer_norm(x)
        x = x.mean(dim=1)   # Global average pooling
        
        
        # Classifier head
        x = self.head(x)

        return x


if __name__ == '__main__':
    # 임의의 ECG 데이터셋 생성
    ecg_data = torch.rand(1, 12, 5000)  # [batch, channels, seq_len]

    # 모델 인스턴스 생성: ECG 데이터에 맞게
    net = MlpMixer(num_channels=12, hidden_dim=32, mlp_token_dim=32, mlp_channel_dim=32, seq_len=5000, num_blocks=2, num_classes=2)

    # 모델 실행
    out = net(ecg_data)

    print(out.shape)  # 예상 출력 형태: [batch, num_classes]