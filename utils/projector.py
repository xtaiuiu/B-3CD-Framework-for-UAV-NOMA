import numpy as np
def proj_generalized_simplex(a, y, r):
    """
    project y onto the generalized simplex <a, x> = r, x >= 0 by using kkt condition
    :param a: numpy array
    :param y: numpy array
    :param r: float
    :return: numpy array
    """
    u = -np.sort(-y/a)
    sorted_idx = np.argsort(-y/a)
    s_a = a[sorted_idx]
    s_a_squared = s_a**2
    z = s_a_squared * u

    rho = 0
    for j in range(len(u)):
        if u[j] + (r - np.sum(z[:j+1])) / (np.sum(s_a_squared[:j+1])) > 0:
            rho = j
        else:
            break
    lam = (r - np.sum(s_a_squared[:rho+1]*u[:rho+1])) / (np.sum(s_a_squared[:rho+1]))
    return np.array([max(0, y[i]+lam*a[i]) for i in range(len(y))])
