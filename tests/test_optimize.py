import numpy as np
from context import variationaloptimization


def test_optimization():
    knapsack = Knapsack()

    x0 = np.ones(10, dtype=np.int)
    minres = variationaloptimization.minimize_variational(knapsack, x0, learning_rate=1e-3,
                                  max_iter=50)
    weight = minres.x.dot(knapsack.weights)
    value = minres.x.dot(knapsack.values)

    assert minres.success
    assert minres.fun == -value
    assert value > 100
    assert value <= 169
    assert weight > 0
    assert weight <= knapsack.max_weight


class Knapsack(object):
    def __init__(self):
        self.values = np.array([30, 12, 62, 43, 27, 50, 18, 22, 24, 30])
        self.weights = np.array([10, 5, 30, 10, 20, 30, 25, 15, 15, 15])
        self.max_weight = 70

    def __call__(self, x):
        value = x.dot(self.values)
        weight = x.dot(self.weights)
        if weight > self.max_weight:
            value = -1e-6
        return -value
