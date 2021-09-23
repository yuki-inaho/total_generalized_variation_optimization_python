import sys
import numpy as np


# TODO: rewrite
def gradient(image):
    """
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    shape = [
        image.ndim,
    ] + list(image.shape)
    gradient = np.zeros(shape, dtype=image.dtype)  # [dim, height, width]
    slice_all = [
        0,
        slice(None, -1),
    ]

    for d in range(image.ndim):
        gradient[tuple(slice_all)] = np.diff(
            image,
            axis=d
        )
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


def divergence(grad):
    div = np.zeros(grad.shape[1:])  # div's shape is [height, width]
    for d in range(grad.shape[0]):  # grad's shape is [dim, height, width]
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(div, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return div


def symmetrized_second_derivative(gradient):
    if gradient.ndim < 3:
        sys.exit("second_derivative: gradient.ndim < 3")
    n_independent_variable = gradient.shape[0]

    
    shape = [n_independent_variable**2,] + list(gradient.shape[1:])
    second_derivative = np.zeros(shape, dtype=gradient.dtype)  # [dim, height, width]

    second_derivative[0, :-1, :] = np.diff(gradient[0], axis=0)
    second_derivative[1, :-1, :] = np.diff(gradient[0], axis=0) 
    second_derivative[1, :, :-1] += np.diff(gradient[1], axis=1)/2
    
    second_derivative[2, :-1, :] = np.diff(gradient[1], axis=0)/2
    second_derivative[2, :, :-1] += np.diff(gradient[0], axis=1)
    
    second_derivative[3, :, :-1] = np.diff(gradient[1], 1)

    """
    for i in range(n_independent_variable):
        slice_all = [
            i * n_independent_variable,
            slice(None, -1),
        ]
        for d in range(n_independent_variable):            
            second_derivative[tuple(slice_all)] = (np.diff(gradient[i], axis=d) + np.diff(gradient[i], axis=n_independent_variable - d -1).T) / 2
            slice_all[0] = i * n_independent_variable + d + 1
            slice_all.insert(1, slice(None))
    """
    
    return second_derivative


def second_order_divergence(second_order_gradient):
    if second_order_gradient.ndim < 3:
        sys.exit("second_order_divergence: second_order_gradient.ndim < 3")
    if second_order_gradient.shape[0] < 4:
        sys.exit("second_order_divergence: second_order_gradient.shape[0] < 4")

    n_independent_variable = int(np.sqrt(second_order_gradient.shape[0]))
    shape = [n_independent_variable,] + list(second_order_gradient.shape[1:])
    res = np.zeros(shape, dtype=second_order_gradient.dtype)

    res[0] = second_order_gradient[0] + second_order_gradient[1].T
    res[1] = second_order_gradient[1] + second_order_gradient[2].T

    """
    for i in range(n_independent_variable):
        for j in range(n_independent_variable):
            d = n_independent_variable * i + j
            this_grad = np.rollaxis(second_order_gradient[d], j)
            this_res = np.rollaxis(res[i], j) 
            this_res[:-1] += this_grad[:-1]
            this_res[1:-1] -= this_grad[:-2]
            this_res[-1] -= this_grad[-2]
    """
    
    return res


def proj_l2(g, lambda_tv=1.0):
    # res = np.copy(g/lambda_tv)
    # denom = np.maximum(1.0, np.sum(np.abs(g), axis=0) / lambda_tv)

    res = np.copy(g)
    denom = np.maximum(np.sqrt(np.sum(g ** 2, 0)) / lambda_tv, 1.0)
    # denom = np.maximum(1.0, np.sum(np.abs(g), axis=0))
    res[0] /= denom
    res[1] /= denom
    return res


def proj_double_norm(u, f, lambda_tv=1.0, tau=1.0):
    return (lambda_tv * u + tau * f) / (lambda_tv + tau)


def norm1(mat):
    return np.sum(np.abs(mat))


def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())


def power_method(data, n_it=100):
    x = data
    for k in range(0, n_it):
        x = -divergence(gradient(x))
        s = np.sqrt(norm2sq(x))
        x /= s
    return np.sqrt(s)