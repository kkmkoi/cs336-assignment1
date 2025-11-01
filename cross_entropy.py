import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失 ℓ_i = - log softmax(o_i)[x_{i+1}]，处理批量维度并返回平均损失。

    参数:
    - logits: (..., vocab_size) 预测 logits
    - targets: (...) 目标 token IDs

    返回:
    - 平均交叉熵损失 (标量)
    """
    # 数值稳定性：减去最大值
    logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]

    # 计算 log_sum_exp
    log_sum_exp = logits_stable.exp().sum(dim=-1).log()

    # 从稳定化后的 logits 中选择目标 logits
    selected_logits = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 计算交叉熵损失
    loss = -(selected_logits - log_sum_exp)
    # perplexity = torch.exp(loss.mean())
    # print(f"困惑度: {perplexity.item()}")
    # 返回批量平均损失
    return loss.mean()