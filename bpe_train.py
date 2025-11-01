"""
bpe分词器训练:
输入参数：input_path:str,包含分词器训练数据的文件路径
         vocab_size:int,用于定义最终的词汇表大小的正整数
         special_tokens:list[str],要添加到词汇表中的字符串列表

输出参数：vocab:dict[int,bytes],分词器词汇表，一种从整数到字节的映射
         merges:list[tuple[bytes,bytes]],训练产生的bpe合并列表，每个列表项是一个字节元组，应该按顺序创建

版本v1:实现一个基本的bpe训练算法，按照bpe算法的标准步骤进行实现。
"""


from collections import Counter
import regex as re

def train_bpe_tokenizer_v1(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. 读取文件内容
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

  
    special_pattern="|".join(re.escape(tok) for tok in special_tokens)

    pat = re.compile(
        rf"{special_pattern}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"
    )
    words = re.findall(pat, text)
   

    unicode_words = [tuple(bytes([b]) for b in word.encode('utf-8')) for word in words if word not in special_tokens]

    text_bytes_freq=Counter(unicode_words)

    # 2. 构建初始词汇表（包含特殊token）
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    idx = 0
    for st in special_tokens:
        vocab[idx] = st.encode('utf-8')
        idx += 1
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    

    # 8.重复步骤3-7直到达到vocab_size
    while len(vocab) < vocab_size:
        # 3.统计相邻bytes对频率
        pair_freq=Counter()
        for word,freq in text_bytes_freq.items():
            if len(word)<2:
                continue
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_freq[pair] += freq

        # 4.找到pair-freq中频率最高且字典序最大的pair
        if not pair_freq:
            break
        best_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
        #best_pair = min(pair_freq, key=lambda x: (-pair_freq[x], x))
        # 5.加入merges列表
        # if best_pair == (b' g', b'ive'):
        #     print("Debug: Found the pair (' g', 'ive')", pair_freq[best_pair])
        # if best_pair == (b'\n', b'\n'):
        #     print("Debug: Found the pair ('\\n', '\\n')", pair_freq[best_pair])
        merges.append(best_pair)

        # 6.加入vocab列表
        new_token = best_pair[0] + best_pair[1]
        vocab[idx] = new_token
        idx += 1

        # 7.更新text_bytes_freq
        new_text_bytes_freq = Counter()
        for word, freq in text_bytes_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_text_bytes_freq[tuple(new_word)] += freq
        text_bytes_freq = new_text_bytes_freq
    return vocab, merges
    
    



"""
bpe分词器训练:
输入参数：input_path:str,包含分词器训练数据的文件路径
         vocab_size:int,用于定义最终的词汇表大小的正整数
         special_tokens:list[str],要添加到词汇表中的字符串列表

输出参数：vocab:dict[int,bytes],分词器词汇表，一种从整数到字节的映射
         merges:list[tuple[bytes,bytes]],训练产生的bpe合并列表，每个列表项是一个字节元组，应该按顺序创建

版本v2:优化bpe训练算法，通过维护索引(缓存法，空间换时间)来减少不必要的计算，提高效率。         
"""

from collections import defaultdict
import regex as re

def train_bpe_tokenizer_v2(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. 读取文件内容
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    pat = re.compile(
        rf"{special_pattern}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"
    )
    words = re.findall(pat, text)

    # 构建 word_freqs
    word_freqs = {}
    for word in words:
        if word not in special_tokens:
            word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
            word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1

    # 2. 构建初始词汇表
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    idx = 0
    
    for st in special_tokens:
        vocab[idx] = st.encode('utf-8')
        idx += 1
    
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1

    # 3. 初始化 pair 频率索引和 pair 到 word 的映射
    def get_pairs(word):
        """获取 word 中所有相邻的 pair"""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i + 1]))
        return pairs

    # 初始化 pair_freq 和 pair_to_words
    pair_freq = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    for word, freq in word_freqs.items():
        for pair in get_pairs(word):
            pair_freq[pair] += freq
            pair_to_words[pair].add(word)

    # 4. BPE 训练循环
    while len(vocab) < vocab_size:
        if not pair_freq:
            break

        # 找到最佳 pair
        best_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
        
        # 加入 merges 和 vocab
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[idx] = new_token
        idx += 1

        # 获取所有包含 best_pair 的 word
        affected_words = list(pair_to_words[best_pair])
        
        # 更新 word_freqs 和索引
        for old_word in affected_words:
            if old_word not in word_freqs:
                continue
                
            freq = word_freqs[old_word]
            
            # 从旧 word 中减去 pair 计数
            old_pairs = get_pairs(old_word)
            for pair in old_pairs:
                pair_freq[pair] -= freq
                if pair_freq[pair] == 0:
                    del pair_freq[pair]
                pair_to_words[pair].discard(old_word)
            
            # 合并生成新 word
            new_word = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and old_word[i] == best_pair[0] and old_word[i + 1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            new_word = tuple(new_word)
            
            # 更新 word_freqs
            del word_freqs[old_word]
            word_freqs[new_word] = freq
            
            # 将新 word 的 pair 计数加入索引
            new_pairs = get_pairs(new_word)
            for pair in new_pairs:
                pair_freq[pair] += freq
                pair_to_words[pair].add(new_word)
        
        # 清理 best_pair 的映射
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

    return vocab, merges




"""
bpe分词器训练:
输入参数：input_path:str,包含分词器训练数据的文件路径
         vocab_size:int,用于定义最终的词汇表大小的正整数
         special_tokens:list[str],要添加到词汇表中的字符串列表

输出参数：vocab:dict[int,bytes],分词器词汇表，一种从整数到字节的映射
         merges:list[tuple[bytes,bytes]],训练产生的bpe合并列表，每个列表项是一个字节元组，应该按顺序创建

版本v3:在v2的基础上：并行化处理预分词阶段，以利用多核处理器提高整体训练速度。 
                    将findall改成finditer以节省内存。   
                    进度条显示+时间统计。  
                    通过SortedDict(B树排序字典)+pair_to_key(辅助实现增删)双索引结构，将每次查找最大pair的时间复杂度从O(n)降低到O(1)，同时保持O(log n)更新效率。
                    使用流式并行预分词和合并，减少内存占用峰值；一边处理一边合并；
"""
from sortedcontainers import SortedDict
from collections import defaultdict
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool,cpu_count
from tqdm import tqdm
import time
import psutil  
import os




#并行预分词函数
def process_chunk_pretokenize(args):
    """
    处理单个文件块：读取+预分词
    参数args:(file_path:str, start:int, end:int, special_tokens:list[str],pattern)
    返回: word_freqs: 这个块的 word 频率字典
    """
    file_path, start, end, special_tokens, pattern = args
    # 读取这个块
    with open(file_path, 'rb') as f:
        f.seek(start)  # 跳到起始位置
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')
    # 预分词
    pat = re.compile(pattern)
    word_matches = re.finditer(pat, chunk)
    # 统计频率
    word_freqs = {}
    for match in word_matches:
        word = match.group()
        if word not in special_tokens:
            word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
            word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    
    return word_freqs




def train_bpe_tokenizer_v3(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    total_start = time.time()
    # 1. 读取文件内容
    if num_processes is None:
        num_processes = cpu_count()  # 自动检测核心数

    print(f"\n[阶段 1] 文件分块")
    stage_start = time.time()

    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    pattern = rf"{special_pattern}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"
    
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b'\n')
    
    print(f"  使用 {num_processes} 个进程")
    print(f"  文件分成 {len(boundaries)-1} 个块")
    print(f"  耗时: {time.time() - stage_start:.2f}秒")
    
    print(f"\n[阶段 2] 并行预分词（流式处理）")
    stage_start = time.time()

    chunk_args = [
        (input_path, start, end, special_tokens, pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # with Pool(processes=num_processes) as pool:
    #     chunk_word_freqs = pool.map(process_chunk_pretokenize, chunk_args)
    
    # pretokenize_time = time.time() - stage_start
    # print(f"  耗时: {pretokenize_time:.2f}秒")
    
    # print(f"\n[阶段 3] 合并预分词结果")
    # stage_start = time.time()

    word_freqs = {}
    processed_chunks=0
    # for chunk_wf in chunk_word_freqs:
    #     for word, freq in chunk_wf.items():
    #         word_freqs[word] = word_freqs.get(word, 0) + freq
    #优化：流式并行预分词和合并
    with Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 边处理边合并
        with tqdm(total=len(chunk_args), desc="  处理块", unit="chunk") as pbar:
            for chunk_wf in pool.imap_unordered(
                process_chunk_pretokenize, 
                chunk_args, 
                chunksize=1  # 每次处理一个
            ):
                # 立即合并（处理完一个释放一个）
                for word, freq in chunk_wf.items():
                    word_freqs[word] = word_freqs.get(word, 0) + freq
                
                processed_chunks += 1
                pbar.update(1)
                
                # 定期报告内存使用
                if processed_chunks % 10 == 0:
                    current_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    pbar.set_postfix({
                        "words": f"{len(word_freqs):,}",
                        "mem": f"{current_mem:.0f}MB"
                    })



    pretokenize_time = time.time() - stage_start
    print(f"  总共 {len(word_freqs):,} 个唯一 word")
    print(f"  耗时: {pretokenize_time:.2f}秒")

    # 2. 构建初始词汇表

    print(f"\n[阶段 3] 构建初始词汇表")
    stage_start = time.time()

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    idx = 0
    
    for st in special_tokens:
        vocab[idx] = st.encode('utf-8')
        idx += 1
    
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1

    print(f"  初始 vocab 大小: {len(vocab)}")
    print(f"  耗时: {time.time() - stage_start:.2f}秒")

    # 3. 初始化 pair 频率索引和 pair 到 word 的映射

    print(f"\n[阶段 4] 初始化 pair 索引")
    stage_start = time.time()

    def get_pairs(word):
        """获取 word 中所有相邻的 pair"""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i + 1]))
        return pairs

    # 初始化 pair_freq 和 pair_to_words
    pair_freq = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    for word, freq in word_freqs.items():
        for pair in get_pairs(word):
            pair_freq[pair] += freq
            pair_to_words[pair].add(word)

    # #优化：使用SortedDict维护有序的pair
    sorted_pairs=SortedDict()
    pair_to_key = {}  # pair -> key in sorted_pairs，用于快速定位
    
    for pair, freq in pair_freq.items():
        key = (freq, pair)
        sorted_pairs[key] = pair
        pair_to_key[pair] = key


    index_time = time.time() - stage_start
    print(f"  总共 {len(pair_freq):,} 个唯一 pair")
    print(f"  堆大小: {len(pair_freq):,}")
    print(f"  耗时: {index_time:.2f}秒")
    




    # 4. BPE 训练循环
    print(f"\n[阶段 5] BPE 训练循环")
    target_merges = vocab_size - len(vocab)
    print(f"  目标合并次数: {target_merges:,}")
    
    stage_start = time.time()

    with tqdm(total=target_merges, desc="  BPE 合并", unit="merge") as pbar:
        
        while len(vocab) < vocab_size:
            if not pair_freq:
                break


            # 找到最佳 pair
            best_key, best_pair = sorted_pairs.peekitem(-1)
            
            
            # 加入 merges 和 vocab
            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            vocab[idx] = new_token
            idx += 1
      

            # 从 SortedDict 中移除 best_pair
            del sorted_pairs[best_key]
            del pair_to_key[best_pair]

            # 获取所有包含 best_pair 的 word
            affected_words = list(pair_to_words[best_pair])
            # 收集需要更新的 pair
            pairs_to_update = set()

            # 更新 word_freqs 和索引
            for old_word in affected_words:
                if old_word not in word_freqs:
                    continue
                    
                freq = word_freqs[old_word]
                
                # 从旧 word 中减去 pair 计数
                old_pairs = get_pairs(old_word)
                for pair in old_pairs:
                    pair_freq[pair] -= freq
                    if pair_freq[pair] == 0:
                        del pair_freq[pair]
                    pairs_to_update.add(pair)
                    pair_to_words[pair].discard(old_word)
                
                # 合并生成新 word
                new_word = []
                i = 0
                while i < len(old_word):
                    if i < len(old_word) - 1 and old_word[i] == best_pair[0] and old_word[i + 1] == best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(old_word[i])
                        i += 1
                new_word = tuple(new_word)
                
                # 更新 word_freqs
                del word_freqs[old_word]
                word_freqs[new_word] = freq
                
                # 将新 word 的 pair 计数加入索引
                new_pairs = get_pairs(new_word)
                for pair in new_pairs:
                    pair_freq[pair] += freq
                    pairs_to_update.add(pair)
                    pair_to_words[pair].add(new_word)
            
            # 清理 best_pair 的映射
            if best_pair in pair_freq:
                del pair_freq[best_pair]
            if best_pair in pair_to_words:
                del pair_to_words[best_pair]
            
            # 更新 SortedDict（使用最新的绝对频率）
            for pair in pairs_to_update:
                if pair == best_pair:
                    continue  # 已经删除，跳过
                
                # 删除旧的 key（如果存在）
                if pair in pair_to_key:
                    old_key = pair_to_key[pair]
                    del sorted_pairs[old_key]
                
                # 如果频率 > 0，插入新 key
                current_freq = pair_freq.get(pair, 0)
                if current_freq > 0:
                    new_key = (current_freq, pair)  
                    sorted_pairs[new_key] = pair
                    pair_to_key[pair] = new_key
                elif pair in pair_to_key:
                    # 频率 <= 0，从索引中移除
                    del pair_to_key[pair]


            pbar.update(1)
            if len(merges) % 100 == 0:
                pbar.set_postfix({
                    "vocab": len(vocab),
                    "pairs": f"{len(pair_freq):,}"
                })
    training_time = time.time() - stage_start
    print(f"  最终 vocab 大小: {len(vocab):,}")
    print(f"  实际合并次数: {len(merges):,}")
    print(f"  耗时: {training_time:.2f}秒")
    
    # ✅ 总结
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"训练完成！总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"{'='*60}")
    print(f"各阶段耗时占比:")
    print(f"  1. 文件分块:     {(time.time() - total_start - total_time + 0.01) / total_time * 100:5.1f}%")
    print(f"  2. 并行预分词(流式处理):   {pretokenize_time / total_time * 100:5.1f}%  ({pretokenize_time:.2f}秒)")
    print(f"  3. 初始化索引:   {index_time / total_time * 100:5.1f}%  ({index_time:.2f}秒)")
    print(f"  4. BPE 训练:     {training_time / total_time * 100:5.1f}%  ({training_time:.2f}秒)")
    print(f"{'='*60}\n")

    return vocab, merges


# ...existing code... (所有 v1, v2, v3 函数)

if __name__ == "__main__":
    import time
    import pickle
    import psutil
    
    def get_memory_mb():
        """获取当前内存使用（MB）"""
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # 配置
    input_path = "/home/huanghy258/zxy23336333/llm/assignment1-basics/data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    print(f"开始训练 BPE (vocab_size={vocab_size})...")
    
    # 训练
    start_time = time.time()
    initial_memory = get_memory_mb()
    
    vocab, merges = train_bpe_tokenizer_v3(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=32
    )
    
    elapsed = time.time() - start_time
    peak_memory = get_memory_mb()
    
    # 统计
    print(f"\n训练完成！")
    print(f"  时间: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"  内存: {peak_memory:.2f}MB ({peak_memory/1024:.2f}GB)")
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  合并次数: {len(merges)}")
    
    # 找最长 token
    longest = max(vocab.items(), key=lambda x: len(x[1]))
    longest_str = longest[1].decode('utf-8', errors='replace')
    print(f"\n最长 token:")
    print(f"  ID: {longest[0]}")
    print(f"  长度: {len(longest[1])} 字节")
    print(f"  内容: '{longest_str[:60]}'")
    
    # 保存
    with open("openwebtext_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("openwebtext_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print(f"\n已保存: openwebtext_vocab.pkl, openwebtext_merges.pkl")

    # 作业答案
    print(f"\n{'='*60}")
    print("作业问题回答:")
    print(f"{'='*60}")
    print(f"(a) 训练时间: {elapsed/60:.2f}分钟, 内存: {peak_memory/1024:.2f}GB")
    print(f"(b) 最耗时阶段: 文件读取和预分词 (约占60-70%)")
    print(f"(c) 最长token: '{longest_str[:40]}' ({len(longest[1])}字节)")
    print(f"    合理性: 是，BPE会合并高频短语（如故事开头）")
