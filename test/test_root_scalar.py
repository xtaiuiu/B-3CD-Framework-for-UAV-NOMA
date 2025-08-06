import numpy as np
from scipy.optimize import root_scalar

a = 1
t = 0.47655089902162434

x0 = 1e-1

f = lambda z: z * np.log1p(a /z) - t
f_prime = lambda z: np.log1p(a / z) - a / (z + a)
f_prime2 = lambda z: -a ** 2 / (z * (z + a) ** 2)

result = root_scalar(f, fprime=f_prime, fprime2=f_prime2, method='halley', x0=x0)

print(f"f = {f(x0)}, f' = {f_prime(x0)}, f'' = {f_prime2(x0)}")
print(result.root)