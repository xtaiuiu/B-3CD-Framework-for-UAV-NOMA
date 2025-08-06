import os
import time

import numpy as np

from benchmark_algorithms.benchmark_OMA.optimize_subproblem_OMA import optimize_p_OMA, optimize_x_OMA
from main_algorithms.optimize_h_theta import optimize_height_theta
from scenarios.scenario_creators import load_scenario


def optimize_p_wrapper_OMA(sc):
    pn, uav, UEs = sc.pn, sc.uav, sc.UEs
    K = len(UEs)
    a, b = np.zeros(K), np.zeros(K)
    for k in range(K):
        a[k] = UEs[k].x
        b[k] = (a[k] * pn.sigma * uav.theta * (uav.h + (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2)
                ** (pn.alpha / 2) / (pn.g_0 * UEs[k].tilde_g))

    return optimize_p_OMA(a, b, pn.p_max)


def optimize_h_theta_wrapper_OMA(sc, eps=1e-6):
    pn, uav, UEs = sc.pn, sc.uav, sc.UEs
    D, W, U, d_array = np.zeros(len(UEs)), np.zeros(len(UEs)), np.zeros(len(UEs)), np.zeros(len(UEs))
    for k in range(len(UEs)):
        D[k] = UEs[k].x
        W[k] = (pn.g_0 * UEs[k].tilde_g * UEs[k].p) / (pn.sigma * UEs[k].x)
        d_array[k] = (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2
    R = pn.radius + np.sqrt(uav.u_x ** 2 + uav.u_y ** 2)
    # return optimize_height_theta(R, d_array, D, W, U, pn.h_min, pn.h_max, pn.theta_min, pn.theta_max,
    #                              pn.alpha / 2, eps=eps)
    f, h, theta = optimize_height_theta(R, d_array, D, W, U, pn.h_min, pn.h_max, pn.theta_min, pn.theta_max,
                                 pn.alpha / 2, eps=eps)
    sc.uav.h, sc.uav.theta = h, theta
    # f_rates = sc.get_UE_rates_OMA()
    return f, h, theta


def optimize_x_wrapper_OMA(sc, t_min, eps=1e-6):
    pn, uav, UEs = sc.pn, sc.uav, sc.UEs
    K = len(UEs)
    a = np.zeros(K)
    for k in range(K):
        a[k] = (pn.g_0 * UEs[k].tilde_g * UEs[k].p) / (
                    pn.sigma * uav.theta * (uav.h + (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2) ** (pn.alpha / 2))

    return optimize_x_OMA(a, pn.b_tot, t_min, eps=1e-4)


def bcd_OMA(sc, tol=1e-4, debug=False):
    t0 = time.perf_counter()
    n, n_max = 0, 10
    f_old = 0
    for n in range(n_max):
        t = time.perf_counter()
        # if n >= 1:
        #     print(f"n = {n}, f = {f}, min_rate = {np.min(sc.get_UE_rates_OMA())}")
        f, h, theta = optimize_h_theta_wrapper_OMA(sc, eps=tol)
        sc.set_h_theta(h, theta)
        # print(
        #         f"n = {n}, f = {f}, optimize h and theta, sqrt_h = {np.sqrt(h)}, sqrt_theta = {np.sqrt(theta) * 180 / np.pi}"
        #         f" time = {time.perf_counter() - t}")
        if (n >= 1) and (abs(f - f_old) < tol or abs((f - f_old) / f) < tol):
            break

        t = time.perf_counter()
        f, x = optimize_x_wrapper_OMA(sc, t_min=f, eps=tol)
        #print(f"n = {n}, f = {f}, optimize x, sum_x = {np.sum(x)}, time = {time.perf_counter() - t}")
        sc.set_x_OMA(x)

        t = time.perf_counter()
        f, p = optimize_p_wrapper_OMA(sc)
        #print(f"n = {n}, f = {f}, optimize p, sum_p = {np.sum(p)}, time = {time.perf_counter() - t}")
        sc.set_p(p)
        f_old = f
    #print(f" bcd OMA finished in {time.perf_counter() - t0} sec")
    return f

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the main_algorithms directory
    main_algorithms_dir = os.path.join(current_dir, '..', '..', 'main_algorithms')
    scenario_path = os.path.join(main_algorithms_dir, 'scenario_5000_UEs.pickle')

    # Load the scenario
    sc = load_scenario(scenario_path)
    sc.reset_scenario_OMA()
    # sc.plot_scenario()
    sc.plot_scenario_kde()
    sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()

    t = time.perf_counter()
    f = bcd_OMA(sc, debug=False, tol=1e-6)
    print(f"f = {f}, BCD finished in {time.perf_counter() - t} sec")
