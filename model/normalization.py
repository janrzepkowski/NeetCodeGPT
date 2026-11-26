import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        eps = 1e-5
        mean = np.mean(x)                       # średnia po cechach
        var = np.mean((x - mean) ** 2)          # wariancja po cechach
        x_hat = (x - mean) / np.sqrt(var + eps) # normalizacja: średnia 0, wariancja ≈ 1
        out = gamma * x_hat + beta              # skala i przesunięcie
        return np.round(out, 5)
