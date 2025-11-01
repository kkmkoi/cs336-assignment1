import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件分块成可以独立计数的部分。
    如果边界最终重叠，可能返回比期望更少的块。
    
    参数:
        file: 二进制文件对象
        desired_num_chunks: 期望的块数量
        split_special_token: 用于分割的特殊标记（字节串形式）
    
    返回:
        块边界位置的列表（字节偏移量）
    """
    assert isinstance(split_special_token, bytes), "必须将特殊标记表示为字节串"

    # 获取文件总大小（字节）
    file.seek(0, os.SEEK_END)  # 移动到文件末尾
    file_size = file.tell()     # 获取当前位置（即文件大小）
    file.seek(0)                # 返回文件开头

    # 计算理想的块大小
    chunk_size = file_size // desired_num_chunks

    # 初始的块边界位置猜测，均匀分布
    # 块从前一个索引开始，不包含最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 最后一个边界必须是文件末尾

    mini_chunk_size = 4096  # 每次向前读取 4KB

    # 调整每个中间边界位置，使其对齐到特殊标记
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]  # 当前边界的初始猜测位置
        file.seek(initial_position)  # 移动到边界猜测位置
        
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块（4KB）

            # 如果到达文件末尾（EOF），将此边界设为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊标记
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到特殊标记，将边界设置在标记位置
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            # 没找到，继续向前搜索下一个 4KB
            initial_position += mini_chunk_size

    # 确保所有边界都是唯一的，但可能少于 desired_num_chunks
    # 使用 set 去重，然后排序
    return sorted(set(chunk_boundaries))


# ## 使用示例
# with open(..., "rb") as f:  # 以二进制模式打开文件
#     num_processes = 4  # 期望分成 4 个块（用于多进程处理）
    
#     # 找到块边界，使用 <|endoftext|> 作为分隔符
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # 以下是串行实现，但你可以通过将每对 start/end 发送给进程集来并行化
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)  # 移动到块的起始位置
        
#         # 读取块内容并解码为 UTF-8 字符串（忽略解码错误）
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
#         # 在你的块上运行预分词，并存储每个预标记的计数
#         # TODO: 在这里添加预分词逻辑