import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    y_train = np.array(y_train)
    unique, index, counts = np.unique(y_train, return_index=True, return_counts=True)
    maxi = max(counts)
    filtered_unique = unique[counts == maxi]
    filtered_index = index[counts == maxi]
    min_index = min(filtered_index)
    
    X_test = np.array(X_test)
    prediction = np.zeros(X_test.shape)
    prediction[:] = y_train[min_index]

    return prediction