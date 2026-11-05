class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x = init
        lr = learning_rate
        g = 0
        for _ in range(iterations):
            g =  2 * x
            x = x - lr * g
        return round(x, 5)
             