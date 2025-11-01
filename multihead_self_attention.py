import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scaled_dot_product_attention import scaled_dot_product_attention
from rope import RotaryPositionEmbedding

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512, use_rope: bool = False, theta: float = 10000.0):
        """
        因果多头自注意力机制
        参数:
            d_model: Transformer 模块输入的维度
            num_heads: 多头注意力机制中使用的头数量
            max_seq_len: 最大序列长度（用于 RoPE 缓存）
            use_rope: 是否使用 RoPE
            theta: RoPE 参数
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # 每个头的维度
        self.use_rope = use_rope

        # 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding(d_model=self.d_head, max_seq_len=max_seq_len, theta=theta)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        前向传播
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码，形状为 (seq_len, seq_len)
        返回:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

      

        # 计算 Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_head)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_head)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_head)
        
        # 应用 RoPE
        if self.use_rope:
            # 将 4维张量展平为 3维张量
            Q = Q.reshape(batch_size * self.num_heads, seq_len, self.d_head)
            K = K.reshape(batch_size * self.num_heads, seq_len, self.d_head)

            # 应用 RoPE
            Q = self.rope(Q)
            K = self.rope(K)

            # 恢复为 4维张量
            Q = Q.view(batch_size, self.num_heads, seq_len, self.d_head)
            K = K.view(batch_size, self.num_heads, seq_len, self.d_head)
        

        # 因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        if mask is not None:
            mask = mask & ~causal_mask
        else:
            mask = ~causal_mask

        # 计算注意力
        attention_output = scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_head)

        # 合并多头
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出投影
        return self.o_proj(attention_output)