import time

import numpy as np
from scipy.optimize import root_scalar

from main_algorithms.optimize_p_benchmark import f_values, f_values_extended

# the function that calculate the achieved data rate of each user
def rate_achieved(a, b, c, x, p):
    K = len(a)
    p_near = p[:K]
    p_far = p[K:]
    rate_near = np.array([x[k]*np.log1p(p_near[k]/a[k]) for k in range(K)])
    rate_far = np.array([x[k]*np.log1p(p_far[k]/(b[k]+p_near[k])) if b[k] > c[k] else x[k]*np.log1p(p_far[k]/(c[k]+p_near[k])) for k in range(K)])
    return np.concatenate((rate_near, rate_far))

# optimization using KKT conditions
# r_k (p_k) = x_k log1p(p_k/a_k)
# r_k^1 (p_k, p_{n(k)}) = x_k log1p(p_k/(b_k + p_{n(k)}))
# r_k^2 (p_k, p_{n(k)}) = x_k log1p(p_k/(c_k + p_{n(k)}))
# n(k) is the index of the near user which is paired with the far user k. Here, n(k) = k - K/2, where K is the number of near users.
def func_t(t, a, b, c, x, P_max):
    val, K = -P_max, len(a)
    x_exp = np.array([np.exp(t / x[k]) - 1 for k in range(K)])  # x_exp[k] = exp(t/x[k] - 1)

    p = np.array([a[k] * x_exp[k] for k in range(K)])
    val += np.sum(p)
    val += np.sum(np.array([(b[k] + p[k]) * x_exp[k] if b[k] > c[k] else (c[k] + p[k]) * x_exp[k] for k in range(K)]))
    return val


def optimize_p_kkt_new(a, b, c, x, P_max, eps=1e-6):
    """
    Optimize p using KKT conditions
    :param a: numpy array of length K
    :param b: numpy array of length K
    :param c: numpy array of length K
    :param x: numpy array of length K
    :param P_max: Total power
    :param eps: rtol for root_scalar
    :return:
    """
    t_min, t_max = 0, 1e8
    for k in range(len(a)):
        t_max = min(t_max, x[k] * np.log1p(P_max / a[k]), x[k] * np.log1p(P_max / b[k]), x[k] * np.log1p(P_max / c[k]))
    if func_t(t_min, a, b, c, x, P_max) * func_t(t_max, a, b, c, x, P_max) > 0:
        print(
            f"t_min = {t_min}, t_max = {t_max}, f(t_min) = {func_t(t_min, a, b, c, x, P_max):.12f}, f(t_max) = {func_t(t_max, a, b, c, x, P_max):.12f}")
        raise ValueError("The signs of func_t at t_min and t_max are the same, cannot use bisect search.")
    root_results = root_scalar(func_t, args=(a, b, c, x, P_max), method='bisect',
                               bracket=[t_min, t_max], x0=t_min, rtol=eps)
    t = root_results.root
    p_near = np.array([a[k] * (np.exp(t / x[k]) - 1) for k in range(len(a))])
    p_far = np.array(
        [(b[k] + p_near[k]) * (np.exp(t / x[k]) - 1) if b[k] > c[k] else (c[k] + p_near[k]) * (np.exp(t / x[k]) - 1) for
         k in range(len(a))])
    p = np.concatenate((p_near, p_far))
    return t, p


if __name__ == '__main__':
    np.set_printoptions(precision=6)
    K = 1500  # number of near users
    a = np.random.choice([1, 2, 5], K)
    b = np.random.choice([1, 2, 5], K)
    c = np.random.choice([1, 2, 5], K)
    x = np.random.choice([1, 2, 5], K)
    P_max = 1000
    start_time = time.perf_counter()
    t, p = optimize_p_kkt_new(a, b, c, x, P_max)
    end_time = time.perf_counter()
    print(f"runtime of optimize_p_kkt_new = {end_time - start_time:.6f} seconds")
    print(f"t = {t}, p = {p}, p_sum = {np.sum(p)}")
    rate = rate_achieved(a, b, c, x, p)
    if not np.allclose(rate, rate[0], rtol=1e-5, atol=1e-8):
        print(f"Standard deviation of rate: {np.std(rate)}")
