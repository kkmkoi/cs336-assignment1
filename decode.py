import torch
import torch.nn.functional as F

def decode(
    model: torch.nn.Module,
    prompt: list[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    end_token: int = None,
    device: str = "cuda:0",
    seq_len: int = 1024  # 模型的最大序列长度
) -> list[int]:
    """
    从语言模型中生成文本。

    参数:
    - model (torch.nn.Module): 训练好的语言模型。
    - prompt (list[int]): 用户提供的提示序列（token ID）。
    - max_tokens (int): 最大生成 token 数。
    - temperature (float): 温度缩放参数。
    - top_p (float): Top-p 采样的阈值。
    - end_token (int): 结束 token 的 ID（例如 <|endoftext|>）。
    - device (str): 设备（如 'cpu' 或 'cuda:0'）。
    - seq_len (int): 模型的最大序列长度。

    返回:
    - list[int]: 生成的 token 序列。
    """
    model.eval()
    generated = prompt[:]
    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_tokens):
        # 如果输入序列长度超过模型的最大序列长度，只保留最近的 seq_len 个 token
        if prompt_tensor.size(1) > seq_len:
            prompt_tensor = prompt_tensor[:, -seq_len:]

        # 前向传播，获取 logits
        with torch.no_grad():
            logits = model(prompt_tensor)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]  # 取最后一个 token 的 logits (1, vocab_size)

        # 温度缩放
        logits = logits / temperature

        # 计算 softmax 概率分布
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)

        # Top-p 采样
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            # 将低概率 token 的概率置为 0
            probs[sorted_indices[sorted_indices_to_remove]] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # 从分布中采样下一个 token
        next_token = torch.multinomial(probs, num_samples=1).item()

        # 将生成的 token 添加到序列中
        generated.append(next_token)

        # 如果遇到结束 token，则停止生成
        if end_token is not None and next_token == end_token:
            break

        # 更新输入序列
        prompt_tensor = torch.tensor([generated], dtype=torch.long, device=device)

    return generated




if __name__ == "__main__":
    # 示例用法
    from transformer_lm import TransformerLM
    from bpe_tokenizer import BPETokenizer

    # 假设我们有一个预训练的模型
    model = TransformerLM(
        vocab_size=10000,
        context_length=1024,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        num_layers=12,
        max_seq_len=1024,
        theta=10000.0,
    )
    checkpoint = torch.load('/home/huanghy258/zxy23336333/llm/assignment1-basics/checkpoint/model_checkpoint_13000.pt', map_location="cuda:6")
    model.load_state_dict(checkpoint["model_state"])
    model.to("cuda:6")
    #prompt是一个txt文本文件，里面存储的是自然语言
    with open('/home/huanghy258/zxy23336333/llm/assignment1-basics/cs336_basics/prompt.txt', 'r') as f:
        prompt_text = f.read()
    tokenizer = BPETokenizer.from_files(
        "/home/huanghy258/zxy23336333/llm/assignment1-basics/tinystories_vocab.pkl",
        "/home/huanghy258/zxy23336333/llm/assignment1-basics/tinystories_merges.pkl"
    )
    prompt_tokens = tokenizer.encode(prompt_text)
    generated_tokens = decode(
        model=model,
        prompt=prompt_tokens,
        max_tokens=100,
        temperature=1.0,
        top_p=0.9,
        device="cuda:6"
    )
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)
