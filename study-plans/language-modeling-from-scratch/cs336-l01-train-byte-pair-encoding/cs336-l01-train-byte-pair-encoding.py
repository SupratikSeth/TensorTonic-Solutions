def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """

    vocab = {id: [id] for id in range(256)}
    encoded = [list(text.encode()) for text in corpus]
    merge = []

    for id in range(256, vocab_size, 1):
        
        dictionary = {}
        for binarySt in encoded:
            for pair in zip(binarySt, binarySt[1:]):
                left, right = pair
                freq = dictionary.get((left, right), 0) + 1
                dictionary[(left, right)] = freq
    
        if dictionary:
            left, right = max(dictionary, key=lambda k: (dictionary[k], vocab[k[0]], vocab[k[1]]))
            vocab[id] = vocab[left] + vocab[right]
            merge.append([left, right, id])
        else:
            break
    
        updated = []
        for binarySt in encoded:
            index = 0
            temp = []
            while index < len(binarySt):
                if index+1 < len(binarySt) and binarySt[index] == left and binarySt[index+1] == right:
                    temp.append(id)
                    index += 2
                else:
                    temp.append(binarySt[index])
                    index += 1
    
            updated.append(temp)
    
        encoded = updated

    learned_vocab = [[i, vocab[i]] for i in range(256, len(vocab), 1)]
    return {
        "vocab": learned_vocab,
        "merges": merge
    }