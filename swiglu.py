import torch
from torch import nn


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络实现:
    FFN(x) = W2 * (SiLU(W1 * x) ⊙ (W3 * x))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        参数:
            d_model: 输入和输出的维度
            d_ff: 内部前馈层的维度，默认为 8/3 * d_model，且为 64 的倍数
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if d_ff is None:
            d_ff = max(64, 8 * d_model // 3 // 64 * 64)  # 确保 d_ff 是 64 的倍数

        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}
        # 定义三层线性变换
        self.w1 = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        返回: 输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 计算 SiLU(W1 * x) 和 W3 * x
        x1 = torch.nn.functional.silu(self.w1(x))
        x2 = self.w3(x)
        # 逐元素相乘 (GLU)
        gated = x1 * x2
        # 输出层 W2
        return self.w2(gated)