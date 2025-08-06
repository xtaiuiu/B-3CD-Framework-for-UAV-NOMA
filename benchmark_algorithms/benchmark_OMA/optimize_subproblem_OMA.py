import time
import warnings

import numpy as np
from scipy.optimize import root_scalar

from scenarios.scenario_creators import load_scenario


def optimize_p_OMA(a, b, P_max):
    # optimize \min_k f_k = a_k * log1p(p_k/b_k) under constraints \sum_k p_k = P_max
    K = len(a)
    f = lambda t: np.sum([b[k] * (np.exp(t / a[k]) - 1) for k in range(K)]) - P_max
    t_min, t_max = 0, np.max([a[k] * np.log1p(P_max / b[k]) for k in range(K)])
    assert f(t_max) + 1e-6 >= (K-1)*P_max, f"f(t_max) = {f(t_max)}, (K-1)*P_max = {(K-1)*P_max}"
    assert t_min < t_max, f"t_min = {t_min}, t_max = {t_max}"
    assert f(t_min) * f(t_max) < 0, f"f_min = {f(t_min)}, f_max = {f(t_max)}"
    t_opt = root_scalar(f, bracket=[t_min, t_max]).root
    p_opt = np.array([b[k] * (np.exp(t_opt / a[k]) - 1) for k in range(K)])
    return t_opt, p_opt


def root_xk_OMA(a_k, t, B_tot, x0=1e-4):
    # find the root of f_k(t) = x_k * log1p(a_k/x_k) - t
    # Use the bisection method.
    f = lambda z: z * np.log1p(a_k / z) - t
    assert f(B_tot) >= 0, f"f(B_tot) = {f(B_tot)}"
    x_l = x0
    while f(x_l) > 0:
        x_l /= 2
    return root_scalar(f, x0=x0, bracket=[x_l, B_tot]).root


def optimize_x_OMA(a, B_tot, t_min, eps=1e-8):
    # optimize \min_k f_k = x_k * log1p(a_k/x_k) under constraints \sum_k x_k = B_tot
    K = len(a)
    min_idx = np.argmin(a)
    sorted_idx = np.argsort(a)
    x0 = B_tot/K
    t_max = B_tot * np.log1p(a[min_idx] / B_tot)
    t, x = None, np.zeros(len(a))

    assert t_min < t_max
    while t_max - t_min > eps and abs((t_max - t_min) / t_max) > eps:
        # print(f"t_min = {t_min}, t_max = {t_max}")
        t = (t_min + t_max) / 2
        for k in sorted_idx:
            x[k] = root_xk_OMA(a[k], t, B_tot, x0)
            x0 = x[k]
        if np.sum(x) > B_tot:
            t_max = t
        else:
            t_min = t
    t = t_min
    for k in sorted_idx:
        x[k] = root_xk_OMA(a[k], t, B_tot, x0)
    return t, x

if __name__ == "__main__":
    sc = load_scenario('scenario_5_UEs.pickle')
    sc.reset_scenario_OMA()
    # sc.plot_scenario_kde()
    a = np.array([1, 1, 1])
    B_tot = 1
    t_min = 0
    t1 = time.perf_counter()
    t, x = optimize_x_OMA(a, B_tot, t_min)
    print(f"t = {t}, x = {x}, sum_x = {np.sum(x)}, time = {time.perf_counter() - t1}")
