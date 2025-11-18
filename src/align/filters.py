def is_valid_sentence_pair(sA, sB, max_ratio=3.0, min_len=3):
    if len(sA) < min_len or len(sB) < min_len:
        return False
    ratio = max(len(sA), len(sB)) / (min(len(sA), len(sB)) + 1e-9)
    if ratio > max_ratio:
        return False
    return True