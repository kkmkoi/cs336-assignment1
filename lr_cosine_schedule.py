import math

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    实现带有 warmup 的余弦学习率调度器。

    参数:
    - t (int): 当前迭代步数。
    - alpha_max (float): 最大学习率。
    - alpha_min (float): 最小学习率。
    - T_w (int): warmup 的迭代步数。
    - T_c (int): 余弦退火的迭代步数。

    返回:
    - alpha_t (float): 当前步的学习率。
    """
    if t < T_w:
        # Warm-up阶段
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # 余弦退火阶段
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        # Post-annealing阶段
        return alpha_min