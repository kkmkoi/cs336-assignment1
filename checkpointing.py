import torch
from typing import Union, BinaryIO, IO
import os

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> None:
    """
    保存模型检查点。

    参数:
    - model (torch.nn.Module): 模型对象。
    - optimizer (torch.optim.Optimizer): 优化器对象。
    - iteration (int): 当前的迭代次数。
    - out (str | os.PathLike | BinaryIO | IO[bytes]): 保存检查点的路径或文件对象。
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    加载模型检查点。

    参数:
    - src (str | os.PathLike | BinaryIO | IO[bytes]): 检查点的路径或文件对象。
    - model (torch.nn.Module): 模型对象。
    - optimizer (torch.optim.Optimizer): 优化器对象。

    返回:
    - iteration (int): 加载的迭代次数。
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]