import torch
from torch import nn, Tensor
from transformer_block import TransformerBlock
from embedding import Embedding
from linear import Linear
from rmsnorm import RMSNorm
from softmax import softmax

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        theta: float,
    ):
        """
        Transformer 语言模型
        参数:
            vocab_size: 词表大小
            context_length: 最大上下文长度
            d_model: Transformer 模块输入的维度
            num_heads: 多头自注意力机制中使用的头数量
            d_ff: 前馈网络的隐藏层维度
            num_layers: Transformer 块的数量
            max_seq_len: RoPE 的最大序列长度
            theta: RoPE 的参数
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        # 嵌入层
        self.token_embedding = Embedding(vocab_size, d_model)
       
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
            for _ in range(num_layers)
        ])

        # 输出层
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        参数:
            x: 输入张量，形状为 (batch_size, seq_len)
        返回:
            输出张量，形状为 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # 检查输入长度是否超过上下文长度
        if seq_len > self.context_length:
            raise ValueError(f"输入序列长度 {seq_len} 超过了最大上下文长度 {self.context_length}。")

        # 嵌入层
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)
   
        # Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 归一化和输出层
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits  # (batch_size, seq_len, vocab_size)