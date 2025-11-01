
"""
BPE分词器实现版本: 基于给定的词汇表和合并列表进行分词、编码和解码操作;单进程实现；

"""


import regex as re
from typing import Iterable, Iterator
import pickle


class BPETokenizer:
   
    def __init__(self, vocab, merges, special_tokens=None):
        #raise NotImplementedError
        """
        从给定的词汇表，合并列表，（可选的特殊词汇表）构建分词器
        词汇表vocab:dict[int,bytes];
        合并表merges:list[tuple[bytes,bytes]];
        特殊标记special_tokens:list[str];
        通过在正则表达式中条件性排除<字符解决特殊情况' <|endoftext|>'(不过这种情况不应该出现，所以这只是临时改动为了PASS);
        """
        #存储原始数据
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        #构建反向词汇表
        self.inverse_vocab = {v: k for k, v in vocab.items()}

        #构建merges优先级字典
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

        #处理特殊字符
        self.special_token_to_id={}
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes in self.inverse_vocab:
                token_id = self.inverse_vocab[token_bytes]
            else:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.inverse_vocab[token_bytes] = token_id
            self.special_token_to_id[token] = token_id

        
        if self.special_tokens:
            #构建预分词正则表达式
            base_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            #按长度降序排列（长的优先匹配）
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_tokens)
            self.pattern = f"(?:{special_pattern})|(?:{base_pattern})"
        else:
            self.pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"



    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        #raise NotImplementedError
        """
        (类方法)用于根据序列化的词汇表和合并表（与bpe训练代码输出格式一致）以及可选的特殊标记列表构建并返回分词器；
        序列化的词汇表文件路径vocab_filepath:str;
        序列化的合并表文件路径merges_filepath:str;
        特殊标记列表special_tokens:list[str];
        """
        with open(vocab_filepath,'rb') as f:
            vocab=pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges=pickle.load(f)
        
        return cls(vocab, merges, special_tokens)




    def encode(self, text: str) -> list[int]:
        #raise NotImplementedError
        """
        编码文本为数字ID序列

        """
        token_ids = []
        #使用finditer()预分词
        for match in re.finditer(self.pattern,text):
            pre_token=match.group()
            if pre_token in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[pre_token])
            else:
                token = [bytes([b]) for b in pre_token.encode('utf-8')]
                while len(token)>1:
                    min_priority = float('inf')
                    min_idx = -1

                    for i in range(len(token)-1):
                        pair = (token[i], token[i+1])
                        priority= self.merge_priority.get(pair,float('inf'))
                        if priority < min_priority:
                            min_priority = priority
                            min_idx = i
                    
                    if min_idx == -1:
                        break

                    token=(token[:min_idx] + [token[min_idx]+token[min_idx+1]] + token[min_idx+2:])
                for t in token:
                    token_ids.append(self.inverse_vocab[t])
        return token_ids
 




    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        #raise NotImplementedError
        """
        流式编码（内存高效）
        参数iterable:字符串的可迭代对象
        返回：迭代器，逐个生成token id

        """
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        #raise NotImplementedError
        """
        解码数字ID序列为文本
        参数ids:token id 列表
        返回解码后的字符串
        """
        bytes_sequence=b''.join([self.vocab[token_id] for token_id in ids])
        # errors='replace' 会将无效字节替换为 U+FFFD (�)
        text=bytes_sequence.decode('utf-8',errors='replace')
        return text













#实现多进程编码
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



def encode_chunk_worker(args):
    """
    多进程worker函数：编码单个文本块
    参数:
        args: (tokenizer, text_chunk)
    """
    tokenizer, text_chunk = args
    return tokenizer.encode(text_chunk)



def encode_file_multiprocess(
    tokenizer: BPETokenizer,
    filepath: str,
    num_processes: int = 32,
    special_token: str = "<|endoftext|>"
) -> list[int]:
    """
    使用多进程编码大文件
    
    参数:
        tokenizer: BPE分词器实例
        filepath: 文件路径
        num_processes: 进程数
        special_token: 用于分割的特殊标记
    
    返回:
        token id列表
    """
    num_processes = min(num_processes, cpu_count())
    split_token = special_token.encode('utf-8')
    
    # 读取并分块
    with open(filepath, 'rb') as f:
        boundaries = find_chunk_boundaries(f, 4000, split_token)
        
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
            if chunk_text:
                chunks.append(chunk_text)
    
    print(f"文件分成 {len(chunks)} 块，使用 {num_processes} 个进程编码...")
    
    # 多进程编码
    with Pool(processes=num_processes) as pool:
        # 准备参数：每个进程需要 (tokenizer, chunk) 元组
        args = [(tokenizer, chunk) for chunk in chunks]
        results = list(tqdm(
            pool.imap(encode_chunk_worker, args),
            total=len(chunks),
            desc="  编码进度",
            unit="块"
        ))
    
    # 合并结果
    all_token_ids = []
    for result in results:
        all_token_ids.extend(result)
    
    return all_token_ids


import numpy as np
def encode_and_save_dataset(
    tokenizer: BPETokenizer,
    input_filepath: str,
    output_filepath: str,
    num_processes: int = 32,
    special_token: str = "<|endoftext|>"
):
    """
    编码数据集并保存为 uint16 的 NumPy 数组
    参数:
        tokenizer: BPE分词器实例
        input_filepath: 输入文本文件路径
        output_filepath: 输出.npy文件路径
        num_processes: 进程数
        special_token: 特殊标记
    """
    print(f"\n处理文件: {input_filepath}")
    
    # 多进程编码
    start_time = time.time()
    token_ids = encode_file_multiprocess(tokenizer, input_filepath, num_processes, special_token)
    elapsed_time = time.time() - start_time
    
    dtype = np.uint16
    token_array = np.array(token_ids, dtype=dtype)
    np.save(output_filepath, token_array)
    
    # 统计信息
    file_size_mb = len(token_ids) * token_array.itemsize / (1024 * 1024)
    print(f"  编码时间: {elapsed_time:.2f} 秒")
    print(f"  Token数量: {len(token_ids):,}")
    print(f"  数据类型: {dtype}")
    print(f"  保存文件: {output_filepath} ({file_size_mb:.2f} MB)")














import time
from pretokenization_example import find_chunk_boundaries

if __name__ == "__main__":


    ts_tokenizer = BPETokenizer.from_files(
        "/home/huanghy258/zxy23336333/llm/assignment1-basics/tinystories_vocab.pkl",
        "/home/huanghy258/zxy23336333/llm/assignment1-basics/tinystories_merges.pkl"
    )
    owt_tokenizer = BPETokenizer.from_files(
        "/home/huanghy258/zxy23336333/llm/assignment1-basics/openwebtext_vocab.pkl",
        "/home/huanghy258/zxy23336333/llm/assignment1-basics/openwebtext_merges.pkl"
    )
    # with open("/home/huanghy258/zxy23336333/llm/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt") as f:
    #     ts_text = "<|endoftext|>".join(f.read().split("<|endoftext|>")[:10])
    # with open("/home/huanghy258/zxy23336333/llm/assignment1-basics/data/owt_valid.txt") as f:
    #     owt_text = "<|endoftext|>".join(f.read().split("<|endoftext|>")[:10])
    # # 计算TinyStories的压缩率
    # ts_bytes = len(ts_text.encode('utf-8'))
    # start_time = time.time()
    # ts_tokens = len(ts_tokenizer.encode(ts_text))
    # elapsed_time = time.time() - start_time
    # throughput_bytes_per_sec = ts_bytes / elapsed_time
    # throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)
    # print(f"吞吐量: {throughput_mb_per_sec:.2f} MB/s ({throughput_bytes_per_sec:.0f} bytes/s)")
    # ts_ratio = ts_tokens / ts_bytes
    # print(f"TinyStories tokenizer: {ts_ratio:.4f} tokens/byte ({1/ts_ratio:.4f} bytes/token)")
    
    # # 计算OpenWebText的压缩率
    # owt_bytes = len(owt_text.encode('utf-8'))
    # owt_tokens = len(owt_tokenizer.encode(owt_text))
    # owt_ratio = owt_tokens / owt_bytes
    # print(f"OpenWebText tokenizer: {owt_ratio:.4f} tokens/byte ({1/owt_ratio:.4f} bytes/token)")
    # 数据文件路径
    data_dir = "/home/huanghy258/zxy23336333/llm/assignment1-basics/data"
    output_dir = "/home/huanghy258/zxy23336333/llm/assignment1-basics/tokenized_data"
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # print("\n" + "=" * 70)
    # print("[1] 编码 TinyStories 数据集")
    # print("=" * 70)
    
    # # TinyStories 训练集
    # encode_and_save_dataset(
    #     ts_tokenizer,
    #     f"{data_dir}/TinyStoriesV2-GPT4-train.txt",
    #     f"{output_dir}/tinystories_train.npy",
    #     num_processes=32
    # )
    
    # # TinyStories 验证集
    # encode_and_save_dataset(
    #     ts_tokenizer,
    #     f"{data_dir}/TinyStoriesV2-GPT4-valid.txt",
    #     f"{output_dir}/tinystories_valid.npy",
    #     num_processes=32
    # )
    
    print("\n" + "=" * 70)
    print("[2] 编码 OpenWebText 数据集")
    print("=" * 70)
    
    # OpenWebText 训练集
    encode_and_save_dataset(
        owt_tokenizer,
        f"{data_dir}/owt_train.txt",
        f"{output_dir}/openwebtext_train.npy",
        num_processes=16
    )
    
    # OpenWebText 验证集
    encode_and_save_dataset(
        owt_tokenizer,
        f"{data_dir}/owt_valid.txt",
        f"{output_dir}/openwebtext_valid.npy",
        num_processes=32
    )
    
    print("\n" + "=" * 70)
    print("[3] 验证保存的数据")
    print("=" * 70)
    
    # 验证保存的数据
    for name in ["tinystories_train", "tinystories_valid", "openwebtext_train", "openwebtext_valid"]:
        filepath = f"{output_dir}/{name}.npy"
        data = np.load(filepath)
        print(f"\n{name}:")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  前10个token: {data[:10]}")
        print(f"  文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ 所有数据集编码完成!")
    print("=" * 70)






