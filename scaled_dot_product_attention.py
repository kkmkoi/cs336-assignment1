import torch
from torch import Tensor
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    """
    实现缩放点积注意力机制，使用 torch.einsum。

    参数:
        Q: 查询张量，形状为 (batch_size, ..., seq_len_q, d_k)
        K: 键张量，形状为 (batch_size, ..., seq_len_k, d_k)
        V: 值张量，形状为 (batch_size, ..., seq_len_k, d_v)
        mask: 可选的布尔掩码，形状为 (seq_len_q, seq_len_k)

    返回:
        输出张量，形状为 (batch_size, ..., seq_len_q, d_v)
    """
    # 计算缩放的点积 scores = QK^T / sqrt(d_k)
    d_k = Q.size(-1)
    scores = torch.einsum("...qk,...nk->...qn", Q, K) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))

    # 如果提供了掩码，则将掩码应用到 scores
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 对 scores 应用 softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 计算注意力输出 output = attention_weights @ V
    output = torch.einsum("...qn,...nv->...qv", attention_weights, V)
    return output