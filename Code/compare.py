import numpy as np

def compare_classification(expected, actual):
    if not np.isscalar(actual) and len(actual) == 1 and np.isnan(actual):
        return False
    if not np.isscalar(expected) and not np.isscalar(actual) and len(expected) == 1 and len(actual) == 1:
        expected = expected[0]
        actual == actual[0]
    if len(actual.flatten()) == 1:
        actual = actual.flatten()[0]
    if len(expected.flatten()) == 1:
        expected = expected.flatten()[0]

    if np.isscalar(expected) and np.isscalar(actual):
        return expected == round(actual)
    if np.isscalar(expected):
        if len(actual) == 1:
            return expected == round(actual[0])    
        return expected == np.argmax(actual)
    elif np.isscalar(actual):
        if len(expected) == 1:
            return expected[0] == round(actual)
        return np.argmax(expected) == round(actual)
    if len(expected) != len(actual):
        return False
    return np.argmax(expected) == np.argmax(actual)

def compare_regression(expected, actual, delta):
    if not np.isscalar(expected):
        expected = expected.flatten()
    if not np.isscalar(actual):
        actual = actual.flatten()

    if np.isscalar(expected) and np.isscalar(actual):
        return abs(expected - actual) <= delta
    if np.isscalar(expected):
        if len(actual) > 1:
            return False
        return abs(expected - actual[0]) <= delta
    elif np.isscalar(actual):
        if len(expected) > 1:
            return False
        return abs(expected[0] - actual) <= delta
    n = len(expected)
    if n != len(actual):
        return False
    res = True
    for i in range(0, n):
        res = res and abs(expected[i] - actual[i]) <= delta
    return res