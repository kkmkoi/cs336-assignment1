import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    实现数据加载器，将长序列的 token 数据分割成批量的输入序列和目标序列。
    参数:
    - x (np.ndarray): 一个整数数组，表示 token ID 的序列。
    - batch_size (int): 每个批次的样本数量。
    - context_length (int): 每个样本的序列长度。
    - device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'）。

    返回:
    - inputs (torch.Tensor): 输入序列张量，形状为 (batch_size, context_length)。
    - targets (torch.Tensor): 目标序列张量，形状为 (batch_size, context_length)。
    """
    n = len(x)
    # 随机采样起始位置
    start_indices = np.random.randint(0, n - context_length, size=batch_size)

    # 构造输入序列和目标序列
    inputs = np.stack([x[i:i + context_length] for i in start_indices])
    targets = np.stack([x[i + 1:i + 1 + context_length] for i in start_indices])

    # 转换为 PyTorch 张量并移动到指定设备
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets