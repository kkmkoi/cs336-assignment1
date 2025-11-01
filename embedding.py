"""
嵌入层的简单实现，类似于 torch.nn.Embedding。
"""




import torch
from torch import nn


class Embedding(nn.Module):
     
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        num_embeddings: 词汇表大小 (不同 token 的数量)
        embedding_dim: 每个 token 的嵌入向量维度
        """


        super().__init__()
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:

        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: 形状为 (...,) 的整数张量，表示输入 token 的索引
        返回: 形状为 (..., embedding_dim) 的嵌入向量
        """
        return self.weight[token_ids]