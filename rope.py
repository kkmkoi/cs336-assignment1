import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding
    """

    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        """
        参数:
            d_model: 嵌入向量的维度 (必须是偶数)
            max_seq_len: 最大序列长度
            theta: 控制旋转角度的常数 (默认 10000.0)
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RoPE.")

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 预计算正弦和余弦值
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2) / d_model))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # Outer product
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        返回:
            旋转后的张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) must match initialized d_model ({self.d_model}).")

        # 分离奇数维和偶数维
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 应用旋转
        x_rotated_even = x_even * self.cos[:seq_len] - x_odd * self.sin[:seq_len]
        x_rotated_odd = x_even * self.sin[:seq_len] + x_odd * self.cos[:seq_len]

        # 合并奇数维和偶数维
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
        return x_rotated