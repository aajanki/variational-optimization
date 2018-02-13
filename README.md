# Minimization of functions with binary vector arguments

A Python module that finds the minimum value of a function f(x) and the argument x where the minimum is attained. The argument x is a vector of 0-1 binary variables. Based on variational optimize ideas from [1].

Example:

```python
import variationaloptimization
import numpy as np

# This is the function to minimize
def f(x):
    values = np.array([30, 12, 62, 23])
    weights = np.array([6, 4, 12, 4])
    max_weight = 14

    value = x.dot(values)
    weight = x.dot(weights)
    if weight > max_weight:
        value = -1e-6
    return -value

# An inital guess for the distribution
theta0 = 0.5*np.ones(4)

# Find the minimum value
minres = variationaloptimization.minimize_variational(f, theta0, learning_rate=1e-2, max_iter=1000)
print('The minimum value is {}, the location of the minimum is {}'.format(minres.fun, minres.x))
```

## Requirements

* Python 3.5+
* numpy
* scipy

## Mathematical derivation

See the separate document for the [derivation](docs/binary_variational_optimization.pdf) of the variational optimization algorithm for the binary domain.

## References

[1] Joe Staines, David Barber: [Variational Optimization](https://arxiv.org/abs/1212.4507)