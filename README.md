# Derivative-free function minimization on a binary vector domain

Minimizes a function `f(x)`, where `x` is a vector of 0-1 binary variables, without requiring the derivative. Based on variational optimize ideas from [1].

## Requirements

* Python 3.5+
* numpy
* scipy

## Mathematical derivation

See the separate document for the [derivation](docs/binary_variational_optimization.pdf) of the variational optimization algorithm for the binary domain.

## References

[1] Joe Staines, David Barber: [Variational Optimization](https://arxiv.org/abs/1212.4507)