import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    """
    AdamW优化器实现
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        初始化AdamW优化器

        参数:
        - params: 待优化的参数
        - lr: 学习率
        - betas: 一阶和二阶矩的超参数 (β1, β2)
        - eps: 防止除零的小数值
        - weight_decay: 权重衰减率
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        执行一次参数更新

        参数:
        - closure: 可选的闭包函数，用于重新计算损失
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW 不支持稀疏梯度')

                # 获取超参数
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                # 获取状态字典
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # 更新一阶和二阶矩
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # 更新参数
                denom = corrected_exp_avg_sq.sqrt().add_(eps/math.sqrt(bias_correction2))
                step_size = lr
                p.data.addcdiv_(corrected_exp_avg, denom, value=-step_size)

                # 应用权重衰减
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay*lr)


                
        return loss