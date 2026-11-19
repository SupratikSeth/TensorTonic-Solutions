import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    eps = 1e-9
    n = len(y_train)
    map = {}
    for x, y in zip(X_train, y_train):
        map.setdefault(y, []).append(x)

    prior = {}
    for label, samples in map.items():
        prior[label] = len(samples) / n

    mu = {}
    for label, samples in map.items():
        numFeatures = len(samples[0])
        temp = []
        for i in range(numFeatures):
            temp.append(0)
            
        for sample in samples:
            for i in range(numFeatures):
                temp[i] = temp[i] + sample[i]

        for i in range(numFeatures):
            temp[i] = temp[i] / len(samples)

        mu[label] = temp


    variance = {}
    for label, samples in map.items():
        numFeatures = len(samples[0])
        temp = []
        for i in range(numFeatures):
            temp.append(0)
            
        for sample in samples:
            for i in range(numFeatures):
                temp[i] = temp[i] + (sample[i] - mu[label][i])**2

        for i in range(numFeatures):
            temp[i] = temp[i] / len(samples)

        variance[label] = temp


    predicted = []
    for x in X_test:
        posterior = {}
        for label in prior:
            posterior[label] = prior[label]

            numFeature = len(x)
            sum = 0.0
            for i in range(numFeature):
                sum += -0.5 * math.log(2 * math.pi * variance[label][i] + eps) - ((x[i] - mu[label][i])**2 / (2 * variance[label][i] + eps))

            posterior[label] = posterior[label] + sum

        maxi = float('-inf')
        lbl = None
        for label, prob in posterior.items():
            if prob > maxi:
                lbl = label
                maxi = prob

        predicted.append(lbl)

    return predicted