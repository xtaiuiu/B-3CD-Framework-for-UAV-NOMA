# optimize x by subgradient
import logging

import numpy as np
import cvxpy as cp

from utils.projector import proj_generalized_simplex


def f(x, a, c, B):
    f_val = 1e8
    n = len(x)
    for i in range(n):
        f_val = min(f_val, B[i] * x[i] * np.log1p(a[i] / (x[i] + c[i])))
    return f_val


def f_values(x, a, c, B):
    return np.array([B[i] * x[i] * np.log1p(a[i] / (x[i] + c[i])) for i in range(len(a))])


def g(x, a, c, B):
    n, grad = len(x), np.zeros_like(x)
    for i in range(n):
        if abs(f(x, a, c, B) - B[i] * x[i] * np.log1p(a[i] / (x[i] + c[i]))) < 1e-8:
            grad[i] = B[i] * (np.log1p(a[i] / (x[i] + c[i])) - a[i] * x[i] / (x[i] + a[i] + c[i]) / (x[i] + a[i]))
            break
    return grad


def optimize_x_subgradient(a, c, B, B_tot):
    n, max_iter = len(a), 50000
    x = np.array([B_tot / n / B[i] for i in range(n)])
    i, f_old, f_new = 0, -1e8, f(x, a, c, B)
    while i < max_iter and abs(f_old - f_new) > 1e-8:
        f_old = f_new
        v = 1 / (i / 10 + 1)
        x += v * g(x, a, c, B)
        x = proj_generalized_simplex(B, x, B_tot)
        f_new = f(x, a, c, B)
        print(f"i = {i: 05d}, f_old = {f_old: .8f}, f_new = {f_new: .8f}")
        i += 1
    return f(x, a, c, B), x


if __name__ == '__main__':
    a = np.array([10, 20])
    c = np.array([20, 30])
    B = np.array([1, 2])
    B_tot = 20
    f1, x1 = optimize_x_subgradient(a, c, B, B_tot)
    print(f"f1 = {f1}, x1 = {x1}, values = {f_values(x1, a, c, B)}")
