import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):

    m = len(y)

    for _ in range(n_epochs):
        if method == 'batch':
            y_pred = np.dot(X, weights)
            error = y_pred - y
            gradient = (2/m) * (np.dot(X.T, error))
            weights -= learning_rate * gradient

        elif method == 'stochastic':
            for i in range(m):
                y_pred = np.dot(X[i], weights)
                error = y_pred - y[i]
                gradient = 2 * error * X[i]
                weights -= learning_rate * gradient

        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]
                
                y_pred = np.dot(Xb, weights)
                error = y_pred - yb
                gradient = (2/len(yb)) * (np.dot(Xb.T, error))
                weights -= learning_rate * gradient
        
        else:
            raise ValueError("Invalid method")

    return weights
