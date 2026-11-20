import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    # Write code here
    if y_left == [] and y_right == []:
        return 0
        
    lenLeft = len(y_left)
    lenRight = len(y_right)
    total = lenLeft + lenRight

    y_left = np.array(y_left)
    countLeft = np.unique(y_left, return_counts=True)[1]
    G_left = 1 - np.sum(np.square(countLeft / lenLeft))
    
    y_right = np.array(y_right)
    countRight = np.unique(y_right, return_counts=True)[1]
    G_right = 1 - np.sum(np.square(countRight / lenRight))

    gini = (lenLeft / total * G_left) + (lenRight / total * G_right)

    return gini