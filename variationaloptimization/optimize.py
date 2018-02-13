import numpy as np
import numpy.random as rnd
from scipy.optimize import OptimizeResult


def minimize_variational(f, theta0, learning_rate=1e-3, max_iter=100, disp=False,
                         callback=None, callback_freq=100):
    """Minimize a scalar, 0-1 input function using variational optimization.

    Parameters
    ----------
    f : function
      The function to minimize. Must take one parameter, a numpy array of 0-1
      binary variables. Must return a float or a double.

    theta0 : numpy array, [n_dimensions]
      An initial guess for the probability of how likely each dimension in the
      function input is 1. Used as a starting point for the optimization.

    learning_rate : float, optional (default=1e-3)
      A base learning rate in SGD. The learning rate is adapted using Adam
      during the optimization.

    max_iter : int, optionnal (default=100)
      The number of iterations.

    disp : boolean, optional (default=False)
      If true, will print status messages after each iteration.

    callback : function, optional (default=None)
      A callable that is called periodically to report the progress. Takes
      three parameters: the value of the variational optimization upper bound,
      the gradient and the number of iterations completed. If None (which is
      the default), the callback is never executed. The optimization can be
      stopped early by returning True from the callback.

    callback_freq : int, optional (default=100)
      How often the callback is executed?
    """
    theta = theta0
    moment = np.zeros(theta0.shape)
    v = np.zeros(theta0.shape)
    num_iter = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    while num_iter < max_iter:
        num_iter += 1

        # Take a stochastic gradient descent step.
        grad = _estimate_grad(f, theta)

        # Adapt the learning rate using Adam
        moment = beta1*moment + (1 - beta1)*grad
        moment_sc = moment/(1 - beta1**num_iter)
        v = beta2*v + (1 - beta2)*grad*grad
        v_sc = v/(1 - beta2**num_iter)

        theta_new = theta - learning_rate*moment_sc/(np.sqrt(v_sc) + epsilon)
        theta_new = np.maximum(np.minimum(theta_new, 1 - 1e-6), 1e-6)

        # estimate the upper bound U(theta) at the updated theta
        execute_callback = callback is not None and num_iter % callback_freq == 0
        if disp or execute_callback:
            uval = _estimate_U(f, theta)

        if disp:
            theta_diff = np.linalg.norm(theta_new - theta, ord=2)
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
