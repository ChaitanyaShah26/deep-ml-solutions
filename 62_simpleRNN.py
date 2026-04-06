import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        self.x = x
        self.h = {}
        self.h[-1] = np.zeros((self.hidden_size, 1))
        self.y_pred = {}

        outputs = []

        for t in range(len(x)):
            xt = x[t].reshape(-1, 1)

            self.h[t] = np.tanh(
                np.dot(self.W_xh, xt) +
                np.dot(self.W_hh, self.h[t - 1]) +
                self.b_h
            )

            self.y_pred[t] = np.dot(self.W_hy, self.h[t]) + self.b_y

            outputs.append(self.y_pred[t]) 

        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        T = len(x)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))
        loss = 0

        for t in reversed(range(T)):
            yt = y[t].reshape(-1, 1)
            y_hat = self.y_pred[t]

            loss += 0.5 * np.sum((y_hat - yt) ** 2)

            dy = y_hat - yt

            dW_hy += np.dot(dy, self.h[t].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - self.h[t] ** 2) * dh

            db_h += dh_raw
            dW_xh += np.dot(dh_raw, x[t].reshape(1, -1))
            dW_hh += np.dot(dh_raw, self.h[t - 1].T)

            dh_next = np.dot(self.W_hh.T, dh_raw)

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

        return loss
