import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x, W1, b1, W2, b2, y = map(np.array, (x, W1, b1, W2, b2, y_true))

        # forward
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        loss = np.mean((z2 - y) ** 2)

        # backward
        dz2 = 2 * (z2 - y) / len(y)
        dz1 = (dz2 @ W2) * (z1 > 0)

        r = lambda a: np.round(a, 4).tolist()
        return {
            'loss': round(float(loss), 4),
            'dW1': r(np.outer(dz1, x)),
            'db1': r(dz1),
            'dW2': r(np.outer(dz2, a1)),
            'db2': r(dz2),
        }
