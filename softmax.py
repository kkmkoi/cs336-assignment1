import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    对输入张量的指定维度应用数值稳定的 softmax 操作。

    参数:
        x: 输入张量
        dim: 应用 softmax 的维度

    返回:
        经过 softmax 归一化的张量
    """
    # 减去该维度的最大值以提升数值稳定性
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - max_val
    exp_x = torch.exp(x_stable)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x