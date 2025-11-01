import torch
from torch import nn, Tensor
from multihead_self_attention import CausalMultiHeadSelfAttention
from rmsnorm import RMSNorm
from swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        """
        Transformer 模块
        参数:
            d_model: Transformer 模块输入的维度
            num_heads: 多头自注意力机制中使用的头数量
            d_ff: 前馈网络的隐藏层维度
            max_seq_len: RoPE 的最大序列长度
            theta: RoPE 的参数
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # 初始化多头自注意力
        self.mha = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=True,
            theta=theta,
        )

        # 初始化 RMSNorm 和前馈网络
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        前向传播
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码，形状为 (seq_len, seq_len)
        返回:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 子层 1: 多头自注意力 + 残差连接
        x = x + self.mha(self.norm1(x), mask)

        # 子层 2: SwiGLU 前馈网络 + 残差连接
        x = x + self.ffn(self.norm2(x))

        return x