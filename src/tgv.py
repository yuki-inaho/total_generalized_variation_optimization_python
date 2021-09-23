import numpy as np
from src.operators import (
    stack_2d_array,
    derivative,
    divergence,
    proj_l2,
    proj_double_norm,
    power_method,
    norm2sq,
    norm1,
    second_order_divergence,
    symmetrized_second_derivative,
)
from src.utils import save_image
from tqdm import tqdm
from tqdm import trange
import odl


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
        # p = proj_l2(p + sigma_n * derivative(u_bar), lambda_tv)  # App-LF(139)

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
    #v = 0 * symmetrized_second_derivative(p)
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
        #u = (u + tau_n * ( p_div + lambda_tv * image)) / (1 + tau_n * lambda_tv)
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



def total_variation_l1(image, lambda_tv=10.0, n_it=500, alpha=0.5, L = 12):
    gamma = lambda_tv
    delta = alpha
    mu = 2 * np.sqrt(gamma * delta) / L 

    tau_n = mu / (2 * gamma)
    sigma_n = mu / (2 * delta)
    theta_n = 1 / (1 + mu)

    u = 0 * image
    p = 0 * stack_2d_array(derivative(u))
    q = 0 * image
    u_bar = 0 * u
    theta_n = 1.0

    for _ in trange(0, n_it):
        # Update dual variables
        p = proj_l2(p + sigma_n * stack_2d_array(derivative(u_bar)), lambda_tv)  # App-LF(139)
        q = proj_l2(q + sigma_n * lambda_tv * (u_bar - image), lambda_tv)  # App-LF(139)

        u_old = u.copy()
        # u = (u + tau_n * divergence(p) + tau_n * lambda_tv * image.copy()) / (1.0 + tau_n * lambda_tv)
        u = u + tau_n * divergence(p) - tau_n * lambda_tv * q

        # Update optimizing parameter
        # theta_n = 1 / np.sqrt(1 + 2 * gamma * tau_n)
        # tau_n = theta_n * tau_n
        # sigma_n = sigma_n / theta_n

        # Update primal variables
        u_bar = u + theta_n * (u - u_old)

        fidelity = 0.5 * norm2sq(u - image)
        tv = norm1(stack_2d_array(derivative(u)))
        energy = 1.0 * fidelity + lambda_tv * tv
        tqdm.write(f"energy:{energy}, tv:{tv}, theta: {theta_n}, tau: {tau_n}, sigma: {sigma_n}")

    return u.copy()



def odl_pdhg(image_f):
    U = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[image_f.shape[0], image_f.shape[1]], dtype="float32")
    A = odl.IdentityOperator(U)
    image_u = U.element(image_f)
    data = A(image_u)
    data += odl.phantom.white_noise(A.range) * np.mean(data) * 0.5
    data.show(title="Simulated Data")

    G = odl.Gradient(U, method="forward", pad_mode="symmetric")
    V = G.range

    Dx = odl.PartialDerivative(U, 0, method="backward", pad_mode="symmetric")
    Dy = odl.PartialDerivative(U, 1, method="backward", pad_mode="symmetric")
    E = odl.operator.ProductSpaceOperator([[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
    W = E.range
    domain = odl.ProductSpace(U, V)
    op = odl.BroadcastOperator(
        A * odl.ComponentProjection(domain, 0), odl.ReductionOperator(G, odl.ScalingOperator(V, -1)), E * odl.ComponentProjection(domain, 1)
    )
    f = odl.solvers.ZeroFunctional(domain)
    l2_norm = odl.solvers.L2NormSquared(A.range).translated(data)
    alpha = 1e-1
    beta = 1.0
    l1_norm_1 = alpha * odl.solvers.L1Norm(V)
    l1_norm_2 = alpha * beta * odl.solvers.L1Norm(W)
    g = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)
    op_norm = 1.1 * odl.power_method_opnorm(op)

    niter = 400  # Number of iterations
    tau = 1.0 / op_norm  # Step size for the primal variable
    sigma = 1.0 / op_norm  # Step size for the dual variable
    callback = odl.solvers.CallbackPrintIteration() & odl.solvers.CallbackShow(step=10, indices=0)
    x = op.domain.zero()
    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=callback)
    x[0].show(title="TGV Reconstruction")
    x[1].show(title="Derivatives", force_show=True)