import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMSNorm: y = x / rms(x) * weight, 其中 rms(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        # 增益参数，按照常规初始化为 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)

        # 在最后一维做归一化
        rms = torch.sqrt(torch.mean(x32.pow(2), dim=-1, keepdim=True) + self.eps)
        y32 = x32 / rms
        # 广播乘以可学习增益
        y32 = y32 * self.weight

        return y32.to(in_dtype)