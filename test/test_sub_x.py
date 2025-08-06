import time
import unittest
import numpy as np
import logging

from main_algorithms.optimize_x import optimize_x_kkt


class MyTestCase(unittest.TestCase):
    def test_optimize_x_kkt(self):
        n_repeats, K = 2, np.random.randint(998, 1000)
        for repeat in range(n_repeats):
            # generate a, c, B, B_tot
            a = 100 * np.random.rand(K) + 0.5
            c = 50 * np.random.rand(K) + 1.0
            B = np.random.choice([1, 2, 5], K)
            B_tot = np.random.choice([50, 100, 200])
            eps = 1e-4
            f, x = optimize_x_kkt(a, c, B, B_tot)
            self.assertLess(abs(np.dot(B, x) - B_tot), eps)
            self.assertTrue(np.all(x > 0))

    def test_optimize_x_cvx(self):
        n_repeats, K = 1, np.random.randint(300, 301)
        for repeat in range(n_repeats):
            # generate a, c, B, B_tot
            a = 1000 * (np.random.rand(K) + 0.5)
            c = 500 * (np.random.rand(K) + 1.0)
            B = np.random.choice([1, 2, 5], K)*10
            B_tot = np.random.choice([50, 100, 200])*20
            eps = 1e-4

            t1 = time.perf_counter()
            f, x = optimize_x_kkt(a, c, B, B_tot)
            print(f" kkt time: {time.perf_counter() - t1}")



if __name__ == '__main__':
    unittest.main()
