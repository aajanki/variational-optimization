import numpy as np
import numpy.random as rnd
from scipy.optimize import OptimizeResult


def minimize_variational(f, x0, learning_rate=1e-3, max_iter=100, disp=False,
                         callback=None, callback_freq=100):
    """Minimize a scalar, 0-1 input function using variational optimization."""
    theta = 0.5*np.ones(x0.shape)
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1

        # take a stochastic gradient descent step
        grad = _estimate_grad(f, theta)
        theta_new = np.maximum(np.minimum(theta - learning_rate*grad,
                                          1 - 1e-6), 1e-6)

        

        # estimate the upper bound U(theta) at the updated theta
        execute_callback = num_iter % callback_freq == 0
        if disp or execute_callback:
            uval = _estimate_U(f, theta)

        if disp:
            theta_diff = np.linalg.norm(theta_new - theta, ord=2)

            #print(theta)
            print('U = {}, theta diff = {}'.format(uval, theta_diff))

        if execute_callback:
            force_stop = callback(uval, theta_new, num_iter)
            if force_stop:
                break

        theta = theta_new

    minf, minx = _sample_minimum(f, theta)
    return OptimizeResult(x=minx, success=True, fun=minf, theta=theta, nit=num_iter)


def _estimate_grad(f, theta, num_samples=5):
    s = np.zeros(theta.shape)
    for _ in range(num_samples):
        x = rnd.binomial(np.ones(theta.shape, dtype=np.int), theta)
        fval = f(x)
        grad = (x-1)/(1-theta) + x/theta
        s += fval * grad
    return s/num_samples


def _estimate_U(f, theta, num_samples=5):
    s = 0
    for _ in range(num_samples):
        x = rnd.binomial(np.ones(theta.shape, dtype=np.int), theta)
        s += f(x)
    return s/num_samples


def _sample_minimum(f, theta, num_samples=5):
    bestx = np.zeros(theta.shape)
    best = np.inf
    for _ in range(num_samples):
        x = rnd.binomial(np.ones(theta.shape, dtype=np.int), theta)
        fval = f(x)
        if fval < best:
            best = fval
            bestx = x
    return (best, bestx)
