import numpy as np
from context import variationaloptimization


def test_optimization():
    knapsack = Knapsack()

    theta0 = 0.5*np.ones(4)
    minres = variationaloptimization.minimize_variational(knapsack, theta0,
                                                          learning_rate=1e-2,
                                                          max_iter=1000)
    weight = minres.x.dot(knapsack.weights)
    value = minres.x.dot(knapsack.values)

    assert minres.success
    assert minres.fun == -value
    assert value >= 53
    assert value <= 65
    assert weight > 0
    assert weight <= knapsack.max_weight


class Knapsack(object):
    def __init__(self):
        self.values = np.array([30, 12, 62, 23])
        self.weights = np.array([6, 4, 12, 4])
        self.max_weight = 14

    def __call__(self, x):
        value = x.dot(self.values)
        weight = x.dot(self.weights)
        if weight > self.max_weight:
            value = -1e-6
        return -value
