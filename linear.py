"""
神经网络工具模块: 实现基础的神经网络层
"""

import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    线性变换层 (无偏置): y = xW^T
    参数初始化: 截断正态分布
        μ = 0
        σ² = 2 / (d_in + d_out)
        截断范围: [-3σ, 3σ]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )

        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,  # 下界
            b=3 * std    # 上界
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())

