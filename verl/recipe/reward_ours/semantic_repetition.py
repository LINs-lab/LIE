import numpy as np
from openai import OpenAI
from typing import List


class VLLMEmbeddingModel:
    """使用 vLLM OpenAI-compatible server 的客户端"""
    def __init__(self, base_url="http://10.102.212.39:8000/v1", model_name="embed", 
                 max_retries=3, timeout=120):
        """
        初始化 vLLM Embedding 模型客户端
        
        Args:
            base_url: vLLM OpenAI-compatible server 的 URL
            model_name: 模型名称
            max_retries: 最大重试次数
            timeout: 请求超时时间
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM server 不需要真实的 API key
            timeout=timeout,
            max_retries=max_retries
        )
        self.model_name = model_name
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量获取 embeddings
        
        Args:
            texts: 文本列表
        
        Returns:
            numpy array: embeddings (N, D)
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        
        # 提取 embeddings
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

    def calculate_similarity_batch(self, all_chunks: List[str], chunk_boundaries: List[tuple]) -> tuple:
        """
        批量计算多个 response 的相似度（Chunk-wise Look-back 逻辑）
        """
        if len(all_chunks) == 0:
            return [], [], [], []
        
        # 1. 一次性获取所有 embeddings
        all_embeddings = self.get_embeddings_batch(all_chunks)
        
        # 2. 归一化 (L2 normalization)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        all_embeddings = all_embeddings / norms
        
        # 3. 计算全局相似度矩阵
        global_similarity_matrix = all_embeddings @ all_embeddings.T
        
        # 4. 按边界提取每个 response 的指标
        avg_similarities = []
        std_similarities = []
        max_similarities = []
        high_similarity_flags = []
        
        for start_idx, end_idx in chunk_boundaries:
            n_chunks = end_idx - start_idx
            
            if n_chunks <= 1:
                avg_similarities.append(0.0)
                std_similarities.append(0.0)
                max_similarities.append(0.0)
                high_similarity_flags.append(1)
            else:
                # 提取该 response 的子矩阵 [n_chunks, n_chunks]
                sub_matrix = global_similarity_matrix[start_idx:end_idx, start_idx:end_idx]
                
                # --- 统计部分：保留原有的 Pair-wise 统计用于监控 (上三角) ---
                upper_triangle_mask = np.triu(np.ones((n_chunks, n_chunks), dtype=bool), k=1)
                all_pairs_similarities = sub_matrix[upper_triangle_mask]
                
                avg_similarities.append(float(np.mean(all_pairs_similarities)))
                max_similarities.append(float(np.max(all_pairs_similarities)))
                if n_chunks == 2:
                    std_similarities.append(0.0)
                else:
                    std_similarities.append(float(np.std(all_pairs_similarities)))
                
                # --- 核心惩罚逻辑修改：Chunk-wise Look-back ---
                # 原理：对每个 chunk i (i>0)，看它与前面所有 chunk [0...i-1] 的最大相似度
                repeated_chunks_count = 0
                for i in range(1, n_chunks):
                    # 获取当前第 i 个 chunk 与之前所有 chunk 的相似度向量
                    look_back_similarities = sub_matrix[i, :i] 
                    if np.max(look_back_similarities) > 0.95:
                        repeated_chunks_count += 1
                
                # 计算重复比例：重复的 chunk 数 / 总的可重复位置数(n-1)
                repetition_ratio = repeated_chunks_count / (n_chunks - 1)
                
                # 判定 Flag：比例 >= 0.1 且 至少有 2 个 chunk 重复（防止短文本误伤）
                # 如果你依然想坚持只看比例，可以去掉后半个条件
                if repeated_chunks_count >= 5 and repetition_ratio >= 0.3:
                    high_similarity_flags.append(1)
                else:
                    high_similarity_flags.append(0)
        
        return avg_similarities, std_similarities, max_similarities, high_similarity_flags

    # def calculate_similarity_batch(self, all_chunks: List[str], chunk_boundaries: List[tuple]) -> tuple:
    #     """
    #     批量计算多个 response 的相似度（一次性 embedding，然后按边界计算）
        
    #     Args:
    #         all_chunks: 所有 response 的所有 chunks 拼接在一起
    #         chunk_boundaries: 每个 response 的 chunk 边界 [(start, end), ...]
        
    #     Returns:
    #         tuple: (avg_similarities, std_similarities, max_similarities, high_similarity_flags) - 每个 response 的平均相似度、标准差、最大相似度和高相似度标志列表（如果大于0.7的similarity个数>10则返回1，否则返回0）
    #     """
    #     if len(all_chunks) == 0:
    #         return [], [], [], []
        
    #     # 1. 一次性获取所有 embeddings
    #     all_embeddings = self.get_embeddings_batch(all_chunks)
        
    #     # 2. 归一化 (L2 normalization)
    #     norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    #     norms = np.where(norms == 0, 1, norms)  # 避免除零
    #     all_embeddings = all_embeddings / norms
        
    #     # 3. 计算全局相似度矩阵
    #     global_similarity_matrix = all_embeddings @ all_embeddings.T
        
    #     # 4. 按边界提取每个 response 的相似度
    #     avg_similarities = []
    #     std_similarities = []
    #     max_similarities = []
    #     high_similarity_flags = []
        
    #     for start_idx, end_idx in chunk_boundaries:
    #         n_chunks = end_idx - start_idx
            
    #         if n_chunks <= 1:
    #             avg_similarities.append(0.0)
    #             std_similarities.append(0.0)
    #             high_similarity_flags.append(0)
    #         else:
    #             # 提取该 response 的子矩阵
    #             sub_matrix = global_similarity_matrix[start_idx:end_idx, start_idx:end_idx]
                
    #             # 提取上三角元素
    #             upper_triangle_mask = np.triu(np.ones((n_chunks, n_chunks), dtype=bool), k=1)
    #             similarities = sub_matrix[upper_triangle_mask]
                
    #             # 计算平均值和标准差
    #             avg_similarities.append(float(np.mean(similarities)))
    #             max_similarities.append(float(np.max(similarities)))
    #             if n_chunks == 2:
    #                 std_similarities.append(0.0)
    #             else:
    #                 std_similarities.append(float(np.std(similarities)))
                
    #             # 计算大于0.7的similarity个数，如果大于6个则返回1
    #             count_high_similarity = np.sum(similarities > 0.8)
    #             if count_high_similarity / len(similarities) >= 0.2:
    #                 high_similarity_flags.append(1)
    #             else:
    #                 high_similarity_flags.append(0)
        
    #     return avg_similarities, std_similarities, max_similarities, high_similarity_flags


def split_text_to_words(text):
    """按照 repetition.py 的方式将文本分割成 words（按空格和下划线分割）"""
    words = []
    for segment in text.split():
        words.extend(segment.split('_'))
    return words


def split_text_to_chunks(text, chunk_size=512):
    """将文本按照 chunk_size 个词划分成多个 chunks"""
    words = split_text_to_words(text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        # 将 words 重新组合成文本（用空格连接）
        chunk_text = ' '.join(chunk_words)
        assert chunk_text != ' ' or chunk_text != '', f"chunk_text: {chunk_text}"
        chunks.append(chunk_text)
    # chunks = text.split("\n\n")
    return chunks


def calculate_semantic_repetition(texts, chunk_size=512, model=None, max_workers=None, batch_size=None):
    """
    计算语义重复度（批量处理：一次性 embedding 所有 chunks）
    
    Args:
        texts: 文本列表
        chunk_size: 每个 chunk 的词数（默认 512）
        model: VLLMEmbeddingModel 实例，如果为 None 则创建新实例
        max_workers: 已废弃，保留参数以保持兼容性
        batch_size: 已废弃，保留参数以保持兼容性
    
    Returns:
        tuple: (avg_scores, std_scores, max_scores, high_similarity_flags) - 每个文本的平均相似度、标准差、最大相似度和高相似度标志列表
    """
    if model is None:
        model = VLLMEmbeddingModel()
    
    # 如果没有文本，直接返回
    if len(texts) == 0:
        return [], [], [], []
    
    # 步骤 1: 为每个文本划分 chunks，记录边界
    all_chunks = []
    chunk_boundaries = []
    
    for text in texts:
        start_idx = len(all_chunks)
        chunks = split_text_to_chunks(text, chunk_size)
        all_chunks.extend(chunks)
        end_idx = len(all_chunks)
        chunk_boundaries.append((start_idx, end_idx))
    
    total_chunks = len(all_chunks)
    print(f"总共划分了 {total_chunks} 个 chunks，来自 {len(texts)} 个文本")
    
    # 步骤 2: 批量处理 - 一次性 embedding 所有 chunks
    try:
        avg_scores, std_scores, max_scores, high_similarity_flags = model.calculate_similarity_batch(all_chunks, chunk_boundaries)
        return avg_scores, std_scores, max_scores, high_similarity_flags
    except Exception as e:
        print(f"批量处理失败: {e}")
        return [0.0] * len(texts), [0.0] * len(texts), [0.0] * len(texts), [0] * len(texts)


if __name__ == "__main__":
    import time
    
    # 测试文本
    test_texts = [
        "This is a test sentence. This is another test sentence. " * 100,  # 重复文本
        "The quick brown fox jumps over the lazy dog. " * 100,  # 另一个重复文本
        "Hello world. How are you today? I am fine thank you. " * 100,  # 第三个文本
    ]
    
    print("=== 测试批量语义重复度计算 ===\n")
    
    # 初始化模型
    print("初始化 vLLM Embedding 模型...")
    model = VLLMEmbeddingModel(
        base_url="http://100.97.236.22:8000/v1",
        model_name="embed",
        timeout=120
    )
    
    # 计算语义重复度
    print(f"\n处理 {len(test_texts)} 个文本...")
    start_time = time.time()
    
    avg_scores, std_scores, max_scores, high_similarity_flags = calculate_semantic_repetition(
        texts=test_texts,
        chunk_size=512,
        model=model
    )
    
    elapsed_time = time.time() - start_time
    
    # 输出结果
    print(f"\n计算完成，耗时: {elapsed_time:.2f} 秒\n")
    print("结果：")
    for i, (avg, std, max_val, flag) in enumerate(zip(avg_scores, std_scores, max_scores, high_similarity_flags)):
        print(f"  文本 {i+1}: avg_similarity = {avg:.4f}, std_similarity = {std:.4f}, max_similarity = {max_val:.4f}, high_similarity_flag = {flag}")
    
    print("\n测试完成！")
