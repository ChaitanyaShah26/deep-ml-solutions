import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features: np.ndarray, labels: np.ndarray,
                 initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int):

    weights = initial_weights.astype(float)
    bias = float(initial_bias)

    mse_values = []
    n = len(features)

    for epoch in range(epochs):

        z = np.dot(features, weights) + bias
        preds = sigmoid(z)

        errors = preds - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(mse, 4))

        d_preds = 2 * errors / n

        d_sigmoid = preds * (1 - preds)

        dz = d_preds * d_sigmoid

        dw = np.dot(features.T, dz)
        db = np.sum(dz)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)

    return updated_weights, updated_bias, mse_values
