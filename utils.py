import numpy as np


def demographic_parity(y_logits, u):
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])


def equalized_odds(y, y_logits, u):
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g = np.zeros([2, 2])
    uc = np.zeros([2, 2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[int(y[i])][1] += y_[i]
            uc[int(y[i])][1] += 1
        else:
            g[int(y[i])][0] += y_[i]
            uc[int(y[i])][0] += 1
    g = g / uc
    return np.abs(g[0, 1] - g[0, 0]) + np.abs(g[1, 1] - g[1, 0])


def equalizied_opportunity(y, y_logits, u):
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if y[i] < 0.999:
            continue
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])


def accuracy(y, y_logits):
    y_ = (y_logits > 0.0).astype(np.float32)
    return np.mean((y_ == y).astype(np.float32))
