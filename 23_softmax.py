import numpy as np

def softmax(scores: list[float]) -> list[float]:
    softmax_values = []

    exp_values = [np.exp(s) for s in scores]

    for s in scores:
        z = (np.exp(s))/(np.sum(exp_values))
        softmax_values.append(z)

    return softmax_values
