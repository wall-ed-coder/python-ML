import numpy as np


def batch_generator(X, y, batch_size):
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    x, Y = X[perm], y[perm]
    for i in range(len(X) // batch_size):
        yield (x[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size])
    if len(X) / batch_size != 0:
        yield (x[len(x) - len(X) % batch_size:], Y[len(x) - len(X) % batch_size:])
