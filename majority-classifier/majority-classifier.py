import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    map = {}
    y_train = np.array(y_train)
    for y in y_train:
        map[y] = map.get(y, 0) + 1

    lbl = -1
    maxi = 0
    for label, count in map.items():
        if count > maxi:
            lbl = label
            maxi = count
    X_test = np.array(X_test)
    prediction = np.zeros(X_test.shape)
    prediction[:] = lbl

    return prediction