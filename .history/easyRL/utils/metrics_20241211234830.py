import numpy as np

def moving_average(data, alpha=0.1):
    ma = []
    for i, val in enumerate(data):
        if i == 0:
            ma.append(val)
        else:
            ma.append(ma[-1] * (1 - alpha) + val * alpha)
    return ma