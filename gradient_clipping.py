import torch
import math

def gradient_clipping(parameters: list, max_norm: float, eps: float = 1e-6) -> None:
    """
    实现梯度裁剪。

    参数:
    - parameters (list): 模型的参数列表，每个参数包含梯度。
    - max_norm (float): 梯度的最大 ℓ2 范数。
    - eps (float): 数值稳定性的小值，默认为 1e-6。

    返回:
    - None: 就地修改参数的梯度。
    """
    # 计算所有参数梯度的总 ℓ2 范数
    total_norm = math.sqrt(sum(p.grad.norm(2).item() ** 2 for p in parameters if p.grad is not None) + eps)

    # 如果总范数超过 max_norm，则缩放梯度
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale_factor)