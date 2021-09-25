import numpy as np
from src.operators import (
    stack_2d_array,
    derivative,
    divergence,
    proj_l2,
    proj_double_norm,
    norm2sq,
    norm1,
    second_order_divergence,
    symmetrized_second_derivative,
)
from tqdm import tqdm
from tqdm import trange


def alg3(image, lambda_tv=10.0, alpha = 0.05, L = 24, n_it=500):
    u = 0 * image
    p = 0 * stack_2d_array(derivative(u))
    u_bar = 0 * u

    gamma = lambda_tv
    delta = alpha
    mu = 2 * np.sqrt(gamma * delta) / L 

    tau_n = mu / (2 * gamma)
    sigma_n = mu / (2 * delta)

    #theta_n = 1 / np.sqrt(1 + 2 * gamma * tau_n)
    theta_n = 1 / (1 + mu)

    for _ in trange(0, n_it):
        # Update dual variables
        u_bar_grad = stack_2d_array(derivative(u_bar))
        p = proj_l2(p + sigma_n * u_bar_grad, 1.0)  # App-LF(139)
        u_old = u.copy()

        p_div = divergence(p)
        u = (u + tau_n * p_div  + tau_n * lambda_tv * image.copy()) / (1.0 + tau_n * lambda_tv)

        # Update optimizing parameter
        theta_n = 1 / np.sqrt(1 + 2 * gamma * tau_n)
        tau_n = theta_n * tau_n
        sigma_n = sigma_n / theta_n

        # Update primal variables
        u_bar = u + theta_n * (u - u_old)

        fidelity = 0.5 * norm2sq(u - image)
        tv = norm1(stack_2d_array(derivative(u)))
        energy = 1.0 * fidelity + lambda_tv * tv
        tqdm.write(f"energy:{energy}, tv:{tv}, theta: {theta_n}, tau: {tau_n}, sigma: {sigma_n}")

    return u.copy()


def tgv_denoise_pd(image, lambda_tv=10.0, alpha = 0.05, L = 24, n_it=500, with_linesearch=False):
    gamma = lambda_tv
    delta = alpha
    mu = 2 * np.sqrt(gamma * delta) / L 

    tau_n = mu / (2 * gamma)
    sigma_n = mu / (2 * delta)

    #theta_n = 1 / np.sqrt(1 + 2 * gamma * tau_n)
    theta_n = 1 / (1 + mu)

    u = 0 * image
    p = 0 * stack_2d_array(derivative(u))
    v = 0 * stack_2d_array(derivative(u))
    q = 0 * stack_2d_array(symmetrized_second_derivative(v))

    u_bar = image
    v_bar = 0 * stack_2d_array(derivative(image))
    for k in trange(0, n_it):
        u_bar_grad = stack_2d_array(derivative(u_bar))
        p = proj_l2(p + sigma_n * (u_bar_grad - v_bar), 2.0)

        v_bar_second_derivative = stack_2d_array(symmetrized_second_derivative(v_bar))
        q = proj_l2(q + sigma_n * v_bar_second_derivative, 1.0)

        u_old = u.copy()
        p_div = divergence(p)
        u = proj_double_norm(u + tau_n * p_div, image, lambda_tv, tau_n)
        u_bar = u + theta_n * (u - u_old)

        v_old = v.copy()
        q_second_div = stack_2d_array(second_order_divergence(q))
        v = v + tau_n * (p + q_second_div)
        v_bar = v + theta_n * (v - v_old)

        fidelity = 0.5 * norm2sq(u - image)
        tv = norm1(stack_2d_array(derivative(u)))
        energy = 1.0 * fidelity + lambda_tv * tv
        tqdm.write(f"energy:{energy}, tv:{tv}, theta: {theta_n}, tau: {tau_n}, sigma: {sigma_n}")

    return u.copy()