import time

import numpy as np
import cvxpy as cp
def optimal_power(n, a_val, b_val, P_tot=1.0, W_tot=1.0):
    # Input parameters: α and β are constants from R_i equation
    n = len(a_val)
    if n != len(b_val):
        print('alpha and beta vectors must have same length!')
        return 'failed', np.nan, np.nan, np.nan

    P = cp.Variable(shape=n)
    W = cp.Variable(shape=n)
    alpha = cp.Parameter(shape=n)
    beta = cp.Parameter(shape=n)
    alpha.value = np.array(a_val)
    beta.value = np.array(b_val)

    # This function will be used as the objective so must be DCP;
    # i.e. elementwise multiplication must occur inside kl_div,
    # not outside otherwise the solver does not know if it is DCP...
    R = cp.kl_div(cp.multiply(alpha, W),
                  cp.multiply(alpha, W + cp.multiply(beta, P))) - \
                  cp.multiply(alpha, cp.multiply(beta, P))

    objective = cp.Minimize(cp.sum(R))
    constraints = [P>=0.0,
                   W>=0.0,
                   cp.sum(P)-P_tot==0.0,
                   cp.sum(W)-W_tot==0.0]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.status, -prob.value, P.value, W.value


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    n = 1000  # number of receivers in the system

    a_val = np.arange(10, n + 10) / (1.0 * n)  # α
    b_val = np.arange(10, n + 10) / (1.0 * n)  # β
    P_tot = 0.5
    W_tot = 1.0
    t1 = time.perf_counter()
    status, utility, power, bandwidth = optimal_power(n, a_val, b_val, P_tot, W_tot)
    print(f" finished in {time.perf_counter() - t1} sec")

    print('Status: {}'.format(status))
    print('Optimal utility value = {:.4g}'.format(utility))
    print('Optimal power level:\n{}'.format(power))
    print('Optimal bandwidth:\n{}'.format(bandwidth))