def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    top_k = set(recommended[:k])
    relevant = set(relevant)
    precesion = len(top_k.intersection(relevant)) / k
    recall = len(top_k.intersection(relevant)) / len(relevant)

    return [precesion, recall]