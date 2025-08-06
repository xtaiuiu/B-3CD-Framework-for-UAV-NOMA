import math
import time

import numpy as np
from scipy.optimize import root_scalar, minimize_scalar


def optimize_single_heihgt(R, d, beta, h_bar_min, h_bar_max, eps=1e-6):
    """
    Minimize arctan(R/sqrt{h})*(h+d)^beta in the interval [h_min, h_bar_max].
    :param R: network radius
    :param d: squared distance of UE_k to the center of the network
    :param beta: alpha/2
    :param h_min: lb
    :param h_bar_max: ub
    :return: the optimal value f^* and optimal solution h^*
    """
    g_k = lambda h: math.atan(R / np.sqrt(h)) * (h + d) ** (beta / 2)
    g_k_prime = lambda h: (beta/2) * math.atan(R / np.sqrt(h)) - R*(h + d)/(2*np.sqrt(h) * (h + R**2))
    g_prime_l, g_prime_h = g_k_prime(h_bar_min), g_k_prime(h_bar_max)
    if g_prime_l * g_prime_h > 0:
        return (g_k(h_bar_max), h_bar_max) if g_prime_l < 0 else (g_k(h_bar_min), h_bar_min)
    else:
        h = root_scalar(g_k_prime, bracket=[h_bar_min, h_bar_max], method='bisect', rtol=eps).root
        #print(f" find h by root_scalar:  g_prime_l = {g_prime_l}, g_prime_h = {g_prime_h}, g_k_l = {g_k(h_bar_min)}, g_k_h = {g_k(h_bar_max)}, g = {g_k(h)}, h = {h}")
        return g_k(h), h
    # # res = minimize_scalar(g_k, bounds=(h_bar_min, h_bar_max), method='bounded')
    # assert res.x >= h_bar_min, f"res.x = {res.x}, h_bar_min = {h_bar_min}"
    # assert res.x <= h_bar_max, f"res.x = {res.x}, h_bar_max = {h_bar_max}"
    # return res.fun, res.x


def optimize_height_theta(R, d_array, D_array, W_array, U_array, h_min, h_max, theta_min, theta_max, beta, eps=1e-6):
    """
    Optimize min_k max_h D_k ln(1 + W_k/(theta(h)*(h + d_k)^beta) + U_k)
    :param R: network radius
    :param d_array: d_k, the squared horizontal distance between UE_k and center of the UAV. Let K be the number of near users.
    Then d_array[:K] are the distance for near users, and d_array[K:2*K] are the distance for far users. d_array[2*K:3*K] are the same
    as d_array[K:2*K], just for computational convenience.
    :param D_array: D_k
    :param W_array: W_k
    :param U_array: U_k
    :param h_min: lb of h
    :param h_max: ub of h
    :param theta_min: lb of theta
    :return: f_opt, h_opt, theta_opt
    """
    K = len(d_array)  # K = 3N, where N is the number of near users
    f_opt, h_opt = 1e8, None

    # f_original = np.zeros(K)
    for k in range(K):
        h_bar_min = max(h_min ** 2, (R / math.tan(theta_max)) ** 2)
        h_bar_max = min(h_max ** 2, (R / math.tan(theta_min)) ** 2)
        assert h_bar_min < h_bar_max, f"h_bar_min = {h_bar_min}, h_bar_max = {h_bar_max}"
        f_k, h_k = optimize_single_heihgt(R, d_array[k], beta, h_bar_min, h_bar_max, eps=eps)
        f_original_k = D_array[k] * np.log1p(W_array[k] / (f_k ** 2 + U_array[k]))
        # f_original[k] = f_original_k
        if f_opt > f_original_k:
            f_opt = f_original_k
            h_opt = h_k
    theta_opt = max(theta_min ** 2, (math.atan(R / (np.sqrt(h_opt)))) ** 2)
    return f_opt, h_opt, theta_opt


if __name__ == '__main__':
    # R, K = 10, 10
    # d_array = np.random.rand(K) + 0.1
    # D_array = 0.5 + np.random.rand(K)
    # W_array = 0.5 + np.random.rand(K)
    # U_array = 0.5 + np.random.rand(K)
    # h_min = 10
    # h_max = 100
    # theta_min = np.random.rand() * (np.pi / 4 - np.pi / 12) + np.pi / 12
    # theta_max = np.pi / 2.5
    # beta = np.random.choice([1, 1.5, 2])
    R = 1
    d_array = np.ones(2)
    D_array = np.ones(2)
    W_array = np.ones(2)
    U_array = np.zeros(2)
    h_min, h_max = 0.1, 10
    theta_min, theta_max = np.pi / 12, np.pi / 2.5
    beta = 1
    f_opt, h_opt, theta_opt = optimize_height_theta(R, d_array, D_array, W_array, U_array, h_min, h_max, theta_min,
                                                    theta_max, beta)
    print(f"f_opt = {f_opt}, h_opt = {h_opt}, theta_opt = {theta_opt}")
