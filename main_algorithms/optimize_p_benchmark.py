# optimize p by subgradient
import copy
import logging
import time

import numpy as np
import cvxpy as cp

from utils.projector import proj_generalized_simplex


def f(p, D0, D1, E1, E2, G0, G1, G2):
    f_val = 1e8
    n = len(D0)
    for i in range(n):
        f_val = min(f_val, D0[i]*np.log1p(p[i]/G0[i]))
    for i in range(n):
        f_val = min(f_val, D1[i]*np.log1p(p[i+n]/(E1[i]*p[i]+G1[i])), D1[i]*np.log1p(p[i+n]/(E2[i]*p[i]+G2[i])))
    return f_val

def f_values(p, D0, D1, E1, E2, G0, G1, G2):
    n = len(D0)
    values = np.zeros(2*n)
    for i in range(n):
        values[i] = D0[i]*np.log1p(p[i]/G0[i])
    for i in range(n):
        values[i+n] = min(D1[i]*np.log1p(p[i+n]/(E1[i]*p[i]+G1[i])), D1[i]*np.log1p(p[i+n]/(E2[i]*p[i]+G2[i])))
    return values


def f_values_extended(p, D0, D1, E1, E2, G0, G1, G2):
    n = len(D0)
    values = np.zeros(3 * n)
    for i in range(n):
        values[i] = D0[i] * np.log1p(p[i] / G0[i])
    for i in range(n):
        values[i + n] = min(D1[i] * np.log1p(p[i + n] / (E1[i] * p[i] + G1[i])),
                            D1[i] * np.log1p(p[i + n] / (E2[i] * p[i] + G2[i])))
    for i in range(n):
        values[i + 2*n] = max(D1[i] * np.log1p(p[i + n] / (E1[i] * p[i] + G1[i])),
                            D1[i] * np.log1p(p[i + n] / (E2[i] * p[i] + G2[i])))
    return values


def g(p, D0, D1, E1, E2, G0, G1, G2):
    n, grad = len(D0), np.zeros_like(p)
    for i in range(n):
        if abs(f(p, D0, D1, E1, E2, G0, G1, G2) - D0[i]*np.log1p(p[i]/G0[i])) < 1e-8:
            grad[i] = D0[i]/(G0[i] + p[i])
            break

    for i in range(n):
        if abs(f(p, D0, D1, E1, E2, G0, G1, G2) - D1[i]*np.log1p(p[i+n]/(E1[i]*p[i]+G1[i]))) < 1e-8:
            grad[i] = -p[i + n] * D1[i] * E1[i] / (E1[i] * p[i] + p[i + n] + G1[i]) / (E1[i] * p[i] + G1[i])
            grad[i+n] = D1[i]/(E1[i]*p[i]+p[i+n]+G1[i])

            break
        if abs(f(p, D0, D1, E1, E2, G0, G1, G2) - D1[i]*np.log1p(p[i+n]/(E2[i]*p[i]+G2[i]))) < 1e-8:
            grad[i] = -p[i + n] * D1[i] * E2[i] / (E2[i] * p[i] + p[i + n] + G2[i]) / (E2[i] * p[i] + G2[i])
            grad[i + n] = D1[i] / (E2[i] * p[i] + p[i + n] + G2[i])
            break

    return grad

def optimize_p_quasisubgradient(D0, D1, E1, E2, G0, G1, G2, P_tot):
    n, max_iter = len(D0), 10000
    n_progress, max_no_progress = 0, 100
    p = np.array([P_tot/n/2 for _ in range(2*n)])
    i, f_old, f_new = 0, 1e8, -f(p, D0, D1, E1, E2, G0, G1, G2)
    f_best, p_best = min(f_old, f_new), p
    gamma, delta, beta, theta = 2, 0.5, 0.9, 2
    f_t, delta_k = f_new, delta
    while i < max_iter and abs((f_old - f_new)/f_new) > 1e-8 and n_progress < max_no_progress:
        f_old = f_new
        #v = 1/(i/200+1)
        # adaptive step size rule
        # f_current = f(p, D0, D1, E1, E2, G0, G1, G2)
        f_t = min(f_t, f_new)
        f_k = f_t - delta_k
        grad = -g(p, D0, D1, E1, E2, G0, G1, G2)
        v = (f_new - f_k) / (max(gamma, np.linalg.norm(grad) ** 2))

        p -= v * grad

        p = proj_generalized_simplex(np.ones_like(p), p, P_tot)

        f_new = -f(p, D0, D1, E1, E2, G0, G1, G2)
        if f_new > f_k:
            # print(f"f(k+1) > f(k)")
            delta_k = max(beta * delta_k, delta)
        else:
            print(f"f(k+1) <= f(k)")
            delta_k = theta * delta_k
        if f_new < f_best:
            n_progress = 0
            f_best = f_new
            p_best = copy.deepcopy(p)
        else:
            n_progress += 1
        if not i % 1:
            print(f"i = {i: 05d}, f_old = {f_old: .8f}, f_new = {f_new: .8f}, f_best = {f_best: .8f}, v = {v}, n_progress = {n_progress}")
        i += 1
    return -f_best, p_best


if __name__ == '__main__':
    np.set_printoptions(precision=6)
    # D0 = np.array([1, 2,])
    # D1 = np.array([2, 3, ])
    # G0 = np.array([2, 3, ])
    # E1 = np.array([1, 2, ])
    # E2 = np.array([2, 1, ])
    # G1 = np.array([2, 1, ])
    # G2 = np.array([1, 1, ])
    # P_tot = 12
    K = 2
    D0 = (np.random.rand(K) + 0.5) * 20
    D1 = (np.random.rand(K) + 0.5) * 20
    G0 = (np.random.rand(K) + 0.5) * 20
    E1 = (np.random.rand(K) + 0.5) * 20
    E2 = (np.random.rand(K) + 0.5) * 20
    G1 = (np.random.rand(K) + 0.5) * 20
    G2 = (np.random.rand(K) + 0.5) * 20
    P_tot = np.random.choice([200, 300, 500])
    t = time.perf_counter()
    f1, x1 = optimize_p_quasisubgradient(D0, D1, E1, E2, G0, G1, G2, P_tot)
    print(f" finished in {time.perf_counter() - t} sec")
    print(f"f1 = {f1}, x1 = {x1}, values = {f_values(x1, D0, D1, E1, E2, G0, G1, G2)}, values_extended = {f_values_extended(x1, D0, D1, E1, E2, G0, G1, G2)}")