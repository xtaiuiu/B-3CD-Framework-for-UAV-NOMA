import time

import numpy as np

from main_algorithms.optimize_h_theta import optimize_height_theta
from main_algorithms.optimize_p import optimize_p_kkt_new
from main_algorithms.optimize_x import optimize_x_kkt, util_achieved_rates
from scenarios.scenario_creators import create_scenario, save_scenario, load_scenario


# These wrapper functions are used to calculate the coefficients of each subproblem.
def optimize_x_wrapper(sc, t_min=0, eps=1e-6):
    pn, uav, UEs = sc.pn, sc.uav, sc.UEs
    K = int(len(sc.UEs) / 2)  # K is the number of near users
    a, c = np.zeros(3 * K), np.zeros(3 * K)
    for k in range(K):  # coefficients of the near users
        a[k] = pn.g_0 * UEs[k].tilde_g * UEs[k].p / (
                pn.sigma * uav.theta * ((UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2 + uav.h) ** (
                    pn.alpha / 2))
    for k in range(K, 2 * K):  # coefficients of the far users of rate 1
        a[k] = pn.g_0 * UEs[k].tilde_g * UEs[k].p / (
                pn.sigma * uav.theta * ((UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2 + uav.h) ** (
                    pn.alpha / 2))
        c[k] = pn.g_0 * UEs[k].tilde_g * UEs[k - K].p / (
                pn.sigma * uav.theta * ((UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2 + uav.h) ** (
                    pn.alpha / 2))
    for k in range(K):  # coefficients of the far users of rate 2
        a[k + 2 * K] = pn.g_0 * UEs[k].tilde_g * UEs[k + K].p / (
                pn.sigma * uav.theta * (
                    (UEs[k - K].loc_x - uav.u_x) ** 2 + (UEs[k - K].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha / 2))
        c[k + 2 * K] = pn.g_0 * UEs[k].tilde_g * UEs[k].p / (
                pn.sigma * uav.theta * (
                    (UEs[k - K].loc_x - uav.u_x) ** 2 + (UEs[k - K].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha / 2))
    f, x = optimize_x_kkt(a, c, pn.b_tot, t_min=t_min, eps=eps)
    rates, rates_extended = util_achieved_rates(a, c, x)
    # print(f"rates = {rates}, rates_extended = {rates_extended}")
    return np.min(rates), x


def optimize_p_wrapper(sc, eps=1e-6):
    pn, uav, UEs = sc.pn, sc.uav, sc.UEs
    K = int(len(sc.UEs) / 2)  # K is the number of near users
    a, b, x = np.zeros(K), np.zeros(K), np.zeros(K)
    for k in range(K):
        x[k] = UEs[k].x
        a[k] = (pn.sigma * uav.theta * UEs[k].x * (
                    (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha / 2)) / (
                       pn.g_0 * UEs[k].tilde_g)
        b[k] = (pn.sigma * uav.theta * UEs[k].x * (
                    (UEs[k + K].loc_x - uav.u_x) ** 2 + (UEs[k + K].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha / 2)) / (
                       pn.g_0 * UEs[k + K].tilde_g)

    t, p = optimize_p_kkt_new(a, b, a, x, pn.p_max, eps=eps)
    return t, p


def optimize_h_theta_wrapper(sc, eps=1e-6):
    pn, uav, UEs = sc.pn, sc.uav, sc.UEs
    K = int(len(sc.UEs) / 2)  # K is the number of near users
    x = np.array([UEs[k].x for k in range(K)])

    c_tmp = pn.g_0 / (pn.sigma * x)  # g_0/ (sigma * x_k)
    W0, W1, W2, U0, U1, U2 = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    d_tmp = np.zeros(3 * K)  # the squared horizontal distance between the k-th UE and the UAV
    # assigning elements to the coefficient arrays
    for k in range(K):
        W0[k] = c_tmp[k] * UEs[k].tilde_g * UEs[k].p

        W1[k] = c_tmp[k] * UEs[k + K].tilde_g * UEs[k + K].p
        U1[k] = c_tmp[k] * UEs[k + K].tilde_g * (UEs[k]).p

        W2[k] = c_tmp[k] * UEs[k].tilde_g * (UEs[k + K]).p
        U2[k] = c_tmp[k] * UEs[k].tilde_g * (UEs[k]).p
        d_tmp[k] = (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2
        d_tmp[k + K] = (UEs[k + K].loc_x - uav.u_x) ** 2 + (UEs[k + K].loc_y - uav.u_y) ** 2
        d_tmp[k + 2 * K] = d_tmp[k + K]
    D = np.concatenate((x, x, x))
    W = np.concatenate((W0, W1, W2))
    U = np.concatenate((U0, U1, U2))
    d_array = d_tmp
    R = pn.radius + np.sqrt(uav.u_x ** 2 + uav.u_y ** 2)

    f, h, theta = optimize_height_theta(R, d_array, D, W, U, pn.h_min, pn.h_max, pn.theta_min, pn.theta_max,
                                        pn.alpha / 2, eps=eps)
    return f, h, theta


def bcd(sc, tol=1e-4, debug=False):
    n, n_max = 0, 10
    f_old = 0

    f_values = []  # store the function values, debug only.

    for n in range(n_max):
        t = time.perf_counter()
        f, h, theta = optimize_h_theta_wrapper(sc, eps=tol)
        sc.set_h_theta(h, theta)
        if debug:
            f_values.append(f)
            print(
                f"n = {n}, f = {f}, min_rate = {np.min(sc.get_UE_rates())}, optimize h and theta, sqrt_h = {np.sqrt(h)}, sqrt_theta = {np.sqrt(theta) * 180 / np.pi}"
                f"time = {time.perf_counter() - t}")
        if (n >= 1) and (abs(f - f_old) < tol or abs((f - f_old)/f) < tol):
            break

        t = time.perf_counter()
        f, p = optimize_p_wrapper(sc, eps=tol)
        if debug:
            f_values.append(f)
            print(f"n = {n}, f = {f}, min_rate = {np.min(sc.get_UE_rates())}, optimize p, sum_p = {np.sum(p)}, time = {time.perf_counter() - t}")
        sc.set_p(p)

        t = time.perf_counter()
        f, x = optimize_x_wrapper(sc, t_min=f, eps=tol)
        if debug:
            f_values.append(f)
            print(f"n = {n}, f = {f}, min_rate = {np.min(sc.get_UE_rates())}, optimize x, sum_x = {np.sum(x)}, time = {time.perf_counter() - t}")
        sc.set_x(x)
        f_old = f
    if debug:
        return f_values
    else:
        return f


if __name__ == "__main__":
    np.random.seed(0)
    # sc = create_scenario(150, 100)
    # save_scenario(sc, 'scenario_150_UEs.pickle')
    sc = load_scenario('../benchmark_algorithms/benchmark_meta_heuristic/scenario_100_UEs.pickle')
    sc.reset_scenario()
    #sc.plot_scenario()

    sc.uav.u_x, sc.uav.u_y = 0, 0
    t = time.perf_counter()
    f = bcd(sc, debug=True, tol=1e-6)
    print(f"f = {f}, BCD finished in {time.perf_counter() - t} sec")
