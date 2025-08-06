# benchmark that use fixed power allocation, only bandwidth and UAV deployment are optimized by BCD algorithm.
import time
import numpy as np
from scenarios.scenario_creators import create_scenario
from main_algorithms.bcd_algorithm import bcd
from main_algorithms.bcd_algorithm import optimize_h_theta_wrapper, optimize_p_wrapper, optimize_x_wrapper

def fixed_power(sc, tol=1e-4, debug=False):
    N = len(sc.UEs)
    powers = np.ones(N) * sc.pn.p_max/N
    sc.set_p(powers)
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
        if (n >= 1) and (abs(f - f_old) < tol or abs((f - f_old) / f) < tol):
            break

        t = time.perf_counter()
        f, x = optimize_x_wrapper(sc, t_min=f, eps=tol)
        if debug:
            f_values.append(f)
            print(
                f"n = {n}, f = {f}, min_rate = {np.min(sc.get_UE_rates())}, optimize x, sum_x = {np.sum(x)}, time = {time.perf_counter() - t}")
        sc.set_x(x)
        f_old = f
    if debug:
        return f_values
    else:
        return f


def fixed_bandwidth(sc, tol=1e-4, debug=False):
    K = int(len(sc.UEs))
    bandwidth = np.ones(K) * sc.pn.b_tot / K
    sc.set_x(bandwidth)
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
        if (n >= 1) and (abs(f - f_old) < tol or abs((f - f_old) / f) < tol):
            break

        t = time.perf_counter()
        f, p = optimize_p_wrapper(sc, eps=tol)
        if debug:
            f_values.append(f)
            print(
                f"n = {n}, f = {f}, min_rate = {np.min(sc.get_UE_rates())}, optimize p, sum_p = {np.sum(p)}, time = {time.perf_counter() - t}")
        sc.set_p(p)
        f_old = f
    if debug:
        return f_values
    else:
        return f


if __name__ == '__main__':
    sc = create_scenario(100, 100)

    sc.reset_scenario()
    sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()
    f_NOMA = bcd(sc, debug=False, tol=1e-6)

    sc.reset_scenario()
    f_fixed_power = fixed_power(sc, debug=False, tol=1e-6)

    sc.reset_scenario()
    f_fixed_band = fixed_bandwidth(sc, debug=False, tol=1e-6)

    print(f"f_NOMA = {f_NOMA}, f_fixed_power = {f_fixed_power}, f_fixed_band = {f_fixed_band}")
