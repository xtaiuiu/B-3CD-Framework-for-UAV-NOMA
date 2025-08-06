import math
import unittest
from main_algorithms.optimize_h_theta import optimize_height_theta, optimize_single_heihgt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_optimize_single_heihgt(self):
        R = 100
        d = 0.2
        beta = 2
        h_min = 0.1
        theta_min = 0.8
        h_bar_max = min(2, (R / np.tan(np.sqrt(theta_min))) ** 2)
        f_k, h_k = optimize_single_heihgt(R, d, beta, h_min, h_bar_max)
        self.assertGreaterEqual(f_k, 0)
        self.assertGreaterEqual(h_k, h_min)
        self.assertGreaterEqual(h_bar_max, h_k)

    def test_optimize_height_theta(self):
        n_repeat, K, R = 10, 10, 100
        for repeat in range(n_repeat):
            # generate d_array, D_arry, W_array, U_array, h_min, h_max, theta_min, beta
            d_array = np.random.rand(K)+0.1
            D_array = 10 + np.random.rand(K)
            W_array = 1 + np.random.rand(K)
            U_array = 0.5 + np.random.rand(K)
            h_min = 10
            h_max = 100
            theta_min = np.pi/12
            theta_max = np.pi/2.5
            beta = np.random.choice([1, 1.5, 2])
            f_opt, h_opt, theta_opt = optimize_height_theta(R, d_array, D_array, W_array, U_array, h_min, h_max, theta_min, theta_max, beta)
            self.assertGreaterEqual(f_opt, 0)
            self.assertGreaterEqual(h_opt, h_min**2)
            self.assertGreaterEqual(h_max**2, h_opt)
            self.assertGreaterEqual(theta_opt, theta_min**2)
            print(f"repeat = {repeat}, f_opt = {f_opt: .10f}, h_opt = {h_opt: .4f}, h_min = {h_min: .4f}, theta_opt = "
                  f"{math.degrees(np.sqrt(theta_opt)): .4f}, theta_min = {math.degrees(theta_min): .4f}")


if __name__ == '__main__':
    unittest.main()

