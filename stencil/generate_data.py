import numpy as np
import random

def generate_data():
    X = [[1 if np.random.normal() > 0 else 0, 1 if np.random.normal() > 0 else 0] for ix in range(100)]
    Y = [1 if c[0] != c[1] else 0 for c in X]
    return np.array(X),np.array(Y)
