import argparse
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from get_batch import get_batch
from checkpointing import save_checkpoint, load_checkpoint
from adamw import AdamW
from lr_cosine_schedule import lr_cosine_schedule
from gradient_clipping import gradient_clipping
from transformer_lm import TransformerLM
from cross_entropy import cross_entropy

def train(args):
    # 设置设备
    device = torch.device(args.device)

    # 加载数据集
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    # 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
    ).to(device)

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # 初始化学习率调度器
    lr_scheduler = lambda t: lr_cosine_schedule(
        t, args.lr, args.lr_min, args.warmup_iters, args.cosine_iters
    )

    # 加载检查点（如果提供了路径）
    start_iteration = 0
    if args.checkpoint_path and args.resume:
        start_iteration = load_checkpoint('/home/huanghy258/zxy23336333/llm/assignment1-basics/checkpoint/model_checkpoint_0.pt', model, optimizer)

    # 开始训练
    model.train()
    for iteration in range(start_iteration, args.max_iters):
        # 获取训练批次
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

        # 前向传播
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        gradient_clipping(model.parameters(), args.max_grad_norm)

        # 参数更新
        optimizer.step()

        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler(iteration)

        # 日志记录
        if iteration % args.log_interval == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")

        # 验证
        if iteration % args.val_interval == 0:
            model.eval()
            val_inputs, val_targets = get_batch(val_data, args.batch_size, args.context_length, device)
            with torch.no_grad():
                val_logits = model(val_inputs)
                val_loss = cross_entropy(val_logits, val_targets)
            print(f"Validation Loss: {val_loss.item():.4f}")
            model.train()

        # 保存检查点
        if iteration % args.checkpoint_interval == 0:
            checkpoint_file = os.path.join(args.checkpoint_path, f"model_checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_file)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--train_data", type=str, default="/home/huanghy258/zxy23336333/llm/assignment1-basics/tokenized_data/tinystories_train.npy", help="Path to training data (np.memmap).")
    parser.add_argument("--val_data", type=str, default="/home/huanghy258/zxy23336333/llm/assignment1-basics/tokenized_data/tinystories_valid.npy", help="Path to validation data (np.memmap).")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length.")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feedforward dimension.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of Transformer layers.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate.")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations.")
    parser.add_argument("--cosine_iters", type=int, default=100000, help="Number of cosine annealing iterations.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max_iters", type=int, default=120000, help="Maximum number of iterations.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval.")
    parser.add_argument("--val_interval", type=int, default=1000, help="Validation interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint save interval.")
    parser.add_argument("--checkpoint_path", type=str, default="/home/huanghy258/zxy23336333/llm/assignment1-basics/checkpoint", help="Path to save/load checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:6", help="Device to train on (e.g., 'cpu', 'cuda:0').")

    args = parser.parse_args()
    train(args)