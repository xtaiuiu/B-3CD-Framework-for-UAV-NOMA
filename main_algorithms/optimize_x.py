import time

from scipy.optimize import root_scalar
import logging


def find_xk_near(a_k, c_k, t, x0_guess=0):
    """
    Find the solution for the near user.
    find the solution for the equation f(x) = x * np.log1p(a_k/(x + c_k)) = t by using bisection search
    :param a_k: a_k
    :param c_k: c_k
    :param t: t
    :return: x
    """
    # assert t < a_k, f"t must strictly lower than a_k: t = {t: .12f}, a_k = {a_k: .12f}"
    f = lambda z: z * np.log1p(a_k / (z + c_k)) - t
    f_prime = lambda z: np.log1p(a_k / (z + c_k)) - a_k * z / (z + a_k + c_k) / (z + a_k)
    f_prime2 = lambda z: -a_k * (a_k * z + 2 * c_k * z + 2 * a_k * c_k + 2 * c_k ** 2) / (
                (z + a_k + c_k) * (z + c_k)) ** 2
    # ub = (a_k ** 2 + c_k ** 2) / (a_k - t) - a_k - c_k
    # ub = ub if ub > 0 else 1e8
    result = root_scalar(f, fprime=f_prime, fprime2=f_prime2, method='halley', x0=x0_guess, rtol=1e-5, )
    # print(f"root scalar iterations: {result.iterations}")
    return result.root


def find_xk_far(a_k, c_k, t, x0_guess=1e-4, rtol=1e-6):
    """
    Find the solution for the far user.
    find the solution for the equation f(x) = x * np.log1p(a_k/(x + c_k)) = t by using bisection search
    :param a_k: should be a two-element array or list
    :param c_k: should be a two-element array or list
    :param t: t
    :return: x
    """
    f = lambda z, i: z * np.log1p(a_k[i] / (z + c_k[i])) - t
    f_prime = lambda z, i: np.log1p(a_k[i] / (z + c_k[i])) - a_k[i] * z / (z + a_k[i] + c_k[i]) / (z + a_k[i])
    f_prime2 = lambda z, i: -a_k[i] * (a_k[i] * z + 2 * c_k[i] * z + 2 * a_k[i] * c_k[i] + 2 * c_k[i] ** 2) / (
                (z + a_k[i] + c_k[i]) * (z + c_k[i])) ** 2
    id_min = np.argmin(a_k)
    x = root_scalar(f, fprime=f_prime, fprime2=f_prime2, method='halley', x0=x0_guess, rtol=rtol, args=(id_min,)).root
    # Find the other two indices (excluding id_min)
    other_indices = [i for i in range(3) if i != id_min]
    if all(f(x, i) > -1e-6 for i in other_indices):
        # f(x, i) > -1e-6 for the other two indices, so x is valid
        return x
    else:
        # Find which of the other indices has the smaller f(x,i) value
        min_f_index = min(other_indices, key=lambda i: f(x, i))
        return root_scalar(f, fprime=f_prime, fprime2=f_prime2, method='halley', x0=x0_guess, rtol=rtol,
                           args=(min_f_index,)).root


def util_total_used_bandwidth(a, c, t):
    """
    util function to evaluate sum_k B_k * f_k^{-1}(t)
    :param a: numpy array
    :param c: numpy array
    :param t: t
    :param eps: tolerance
    :return: the total used bandwidth
    """
    assert t < np.min(a), f"t must strictly lower than min_a: t = {t: .12f}, min_a = {np.min(a): .12f}"
    x = np.zeros_like(a)
    for i in range(len(a)):
        if i < len(a) / 2:
            x[i] = find_xk_near(a[i], c[i], t)
        else:
            x[i] = find_xk_far(a[i], c[i], t)
    return np.dot(B, x)


def util_achieved_rates(a, c, x):
    K_n = int(len(a) / 3)
    rates, rates_extend = np.zeros(K_n * 2), np.zeros(K_n * 3)
    for k in range(len(rates)):
        if k < K_n:
            rates[k] = x[k] * np.log1p(a[k] / (x[k] + c[k]))
            rates_extend[k] = rates[k]
        else:
            rates[k] = min(x[k - K_n] * np.log1p(a[k] / (x[k - K_n] + c[k])),
                           x[k - K_n] * np.log1p(a[k + K_n] / (x[k - K_n] + c[k + K_n])))
            rates_extend[k] = x[k - K_n] * np.log1p(a[k] / (x[k - K_n] + c[k]))
            rates_extend[k + K_n] = x[k - K_n] * np.log1p(a[k + K_n] / (x[k - K_n] + c[k + K_n]))
    return rates, rates_extend


def optimize_x_kkt(a, c, B_tot, t_min=0, eps=1e-6):
    """
    Optimize Sub_x by using bi-section search
    :param a: a numpy array of length K+K/2, where K is the number of users.
    For the first K/2 elements are the near users, the K/2 to K elements are the far users of rate 1,
    the last K/2 elements are the far users of rate 2. The other parameters are the same as a.
    :param c: a numpy array of length K+K/2
    :param B_tot: Total bandwidth
    :param eps: precision
    :return: x_star
    """
    K, max_iter = int(len(a) * 2 / 3), 50  # K is the number of users
    K_near = int(K / 2)  # the number of near users
    t_min = t_min
    id_min = np.argmin(a)
    t_max = B_tot * np.log1p(a[id_min] / (B_tot + c[id_min]))
    t_max_guess = np.max(np.array([B_tot/K_near * 2 * np.log1p(a[k] / (B_tot/K_near * 2 + c[k])) for k in range(K)]))

    assert t_max < np.min(a), f"min(a) = {np.min(a)}, t_max = {t_max}"
    #assert t_max_gess < t_max, f"t_max = {t_max}, t_max_gess = {t_max_gess}"
    t_max = min(t_max, t_max_guess)
    assert t_max + 1e-8 >= t_min, f"t_min = {t_min}, t_max = {t_max}"
    n, x = 0, np.zeros(K_near)

    for n in range(max_iter):
        # print(f"n = {n}, t_min = {t_min}, t_max = {t_max}, t_max_guess = {t_max_guess}")
        t = (t_min + t_max) / 2
        if (abs(t_max - t_min) > 1e-3) and (abs(t_max - t_min) / t_max > 1e-3):
            for k in range(K_near):
                x[k] = find_xk_far(np.array([a[k], a[k + K_near], a[k + 2 * K_near]]),
                                   np.array([c[k], c[k + K_near], c[k + 2 * K_near]]), t, rtol=eps)
            if np.sum(x) < B_tot:
                t_min = t
            else:
                t_max = t
        else:
            break
    if n == max_iter - 1:
        logging.log(logging.WARNING, f"max iter reached, t_max - t_min = {t_max - t_min}")

    return t_min, np.array([find_xk_far(np.array([a[k], a[k + K_near], a[k + 2 * K_near]]),
                                        np.array([c[k], c[k + K_near], c[k + 2 * K_near]]), t_min) for k in
                            range(K_near)])


import numpy as np

if __name__ == '__main__':
    # a = np.array([10, 20])
    # c = np.array([20, 30])
    # B = np.array([1, 2])
    # B_tot = 20
    K = 500  # K is the number of near users
    a = np.random.choice([1, 2], 3 * K)
    c = np.random.choice([1, 2], 3 * K)
    B_tot = 200
    # a = np.ones(K*3)
    # c = np.array([1, 2, 3, 4, 5, 6])
    # B = np.ones(3*K)
    # B_tot = 1
    # t = 0.1
    # t1 = time.perf_counter()
    # for j in range(20):
    #     for k in range(K):
    #         find_xk(a[k], c[k], t)
    # print(f"time = {time.perf_counter() - t1}")
    # print(util_total_used_bandwidth(a, c, B, 0.696091015917038147))
    t = time.perf_counter()
    f, x = optimize_x_kkt(a, c, B_tot)
    print(f" optimize x in {time.perf_counter() - t} seconds")
    t = time.perf_counter()
    print(f" optimize x optimized in {time.perf_counter() - t} seconds")
    # Check if x equals x_optimized
