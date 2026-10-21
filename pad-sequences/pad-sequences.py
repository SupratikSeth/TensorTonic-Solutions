import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = 0
        for seq in seqs:
            max_len = max(max_len, len(seq))

    N = len(seqs)
    if N == 0:
        return np.empty(shape=(0, 0), dtype=np.int64)
        
    padded_sequence = []
    for seq in seqs:
        if len(seq) <= max_len:
            padded_sequence.append(seq + [pad_value] * (max_len - len(seq)))
        else:
            padded_sequence.append(seq[:max_len])

    return np.array(padded_sequence, dtype=np.int64)