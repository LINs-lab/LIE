def calculate_distinct_ngram_ratio(text, n=10):
    """
    使用哈希算法计算 text 中唯一 n-gram 占总 n-gram 的比例。
    参考 repetition.py 的逻辑。
    """
    if not text:
        return 1.0

    # 遵循 repetition.py 的分词逻辑
    words = []
    for segment in text.split():
        words.extend(segment.split('_'))

    if len(words) < n:
        return 1.0

    # 使用 set 记录唯一的哈希值
    distinct_hashes = set()
    total_ngrams = 0

    for i in range(len(words) - n + 1):
        # 提取窗口并计算哈希
        window = tuple(words[i:i+n])
        window_hash = hash(window)

        distinct_hashes.add(window_hash)
        total_ngrams += 1

    if total_ngrams == 0:
        return 1.0

    # 比例 = 唯一哈希数 / 总 n-gram 数
    return len(distinct_hashes) / total_ngrams


def calculate_distinct_ngram(text, n=10):
    """
    使用哈希算法计算 text 中唯一 n-gram 占总 n-gram 的比例。
    参考 repetition.py 的逻辑。
    """
    if not text:
        return 1.0

    # 遵循 repetition.py 的分词逻辑
    words = []
    for segment in text.split():
        words.extend(segment.split('_'))

    if len(words) < n:
        return 1.0

    # 使用 set 记录唯一的哈希值
    distinct_hashes = set()

    for i in range(len(words) - n + 1):
        # 提取窗口并计算哈希
        window = tuple(words[i:i+n])
        window_hash = hash(window)

        distinct_hashes.add(window_hash)

    return len(distinct_hashes)
