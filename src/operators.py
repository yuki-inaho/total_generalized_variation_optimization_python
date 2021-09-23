import sys
import numpy as np
from typing import List


def stack_2d_array(arrays: List[np.ndarray]) -> np.ndarray:
    if len(arrays) == 0:
        return np.asarray([])
    if arrays[0].ndim != 2:
        sys.exit(f"<stack_2d_array>: arrays[0].ndim != 2 ({arrays[0].ndim})")
    arrays_stacked = np.dstack(arrays)  # H:W:C
    return arrays_stacked.transpose((2, 0, 1))  # C:H:W


def derivative(image):
    derivative_y = np.zeros_like(image)
    derivative_x = np.zeros_like(image)
    derivative_y[:-1, :] = image[1:, :] - image[:-1, :]
    derivative_x[:, :-1] = image[:, 1:] - image[:, :-1]
    return derivative_y, derivative_x


def divergence(grad_list: np.ndarray):
    grad_y = grad_list[0]
    grad_x = grad_list[1]

    shape = list(grad_list.shape[1:])
    div = np.zeros(shape, dtype=grad_list.dtype)

    div[:, 0] += grad_x[:, 0]
    div[:, 1:-1] += grad_x[:, 1:-1] - grad_x[:, 0:-2]
    div[:, -1] -= grad_x[:, -1]

    div[0, :] += grad_y[0, :]
    div[1:-1, :] += grad_y[1:-1, :] - grad_y[0:-2, :]
    div[-1, :] -= grad_y[-1, :]
    return div


def symmetrized_second_derivative(grad):
    if grad.ndim < 3:
        sys.exit("<second_derivative>: grad.ndim < 3")
    if grad.shape[0] != 2:
        sys.exit("<second_derivative>: grad.shape[0] != 2")

    grad_y = grad[0]
    grad_x = grad[1]

    grad_yy, grad_yx = derivative(grad_y)
    grad_xy, grad_xx = derivative(grad_x)
    #grad_xy_sym = (grad_xy + grad_yx) / 2

    return grad_yy, grad_yx, grad_xy, grad_xx


def second_order_divergence(second_order_derivative, draw=False):
    if second_order_derivative.ndim != 3:
        sys.exit("<second_order_divergence>: second_order_derivative.ndim < 3")
    if second_order_derivative.shape[0] != 4:
        sys.exit("<second_order_divergence>: second_order_derivative.shape[0] != 4")

    derivative_yy = second_order_derivative[0]
    derivative_yx = second_order_derivative[1]
    derivative_xy = second_order_derivative[2]
    derivative_xx = second_order_derivative[3]

    div_sec_x = np.zeros_like(derivative_xx)
    div_sec_x[:, 0] += derivative_xx[:, 0]
    div_sec_x[:, 1:-1] += derivative_xx[:, 1:-1] - derivative_xx[:, 0:-2]
    div_sec_x[:, -1] -= derivative_xx[:, -1]
    div_sec_x[0, :] += derivative_xy[0, :]
    div_sec_x[1:-1, :] += derivative_xy[1:-1, :] - derivative_xy[0:-2, :]
    div_sec_x[-1, :] -= derivative_xy[-1, :]

    div_sec_y = np.zeros_like(derivative_yx)
    div_sec_y[:, 0] += derivative_yx[:, 0]
    div_sec_y[:, 1:-1] += derivative_yx[:, 1:-1] - derivative_yx[:, 0:-2]
    div_sec_y[:, -1] -= derivative_yx[:, -1]
    div_sec_y[0, :] += derivative_yy[0, :]
    div_sec_y[1:-1, :] += derivative_yy[1:-1, :] - derivative_yy[0:-2, :]
    div_sec_y[-1, :] -= derivative_yy[-1, :]

    return div_sec_y, div_sec_x


def proj_l2(g, alpha=1.0):
    res = np.copy(g)
    denom = np.maximum(1.0, np.abs(g).sum(0) / alpha)
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
        x = -divergence(derivative(x))
        s = np.sqrt(norm2sq(x))
        x /= s
    return np.sqrt(s)
