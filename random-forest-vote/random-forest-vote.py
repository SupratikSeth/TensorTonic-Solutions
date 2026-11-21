def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    # Write code here
    m = len(predictions)
    n = len(predictions[0])

    vote = []

    for i in range(n):
        map = {}
        maxi = 0
        label = 0
        for t in range(m):
            map[predictions[t][i]] = map.get(predictions[t][i], 0) + 1
            if map[predictions[t][i]] > maxi:
                label = predictions[t][i]
                maxi = map[predictions[t][i]]
            elif map[predictions[t][i]] == maxi and predictions[t][i] < label:
                label = predictions[t][i]

        vote.append(label)

    return vote
