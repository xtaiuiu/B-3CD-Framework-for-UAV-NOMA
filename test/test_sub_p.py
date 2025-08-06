import time
import unittest
import numpy as np

from main_algorithms.optimize_p import optimize_p_kkt
from main_algorithms.optimize_p_benchmark import optimize_p_quasisubgradient


class MyTestCase(unittest.TestCase):
    def test_optimize_p_kkt(self):
        n_repeats, K = 1, np.random.randint(100, 101)
        for repeat in range(n_repeats):
            # generate parameters
            D0 = (np.random.rand(K) + 0.5) * 20
            D1 = (np.random.rand(K) + 0.5) * 20
            G0 = (np.random.rand(K) + 0.5) * 20
            E1 = (np.random.rand(K) + 0.5) * 20
            E2 = (np.random.rand(K) + 0.5) * 20
            G1 = (np.random.rand(K) + 0.5) * 20
            G2 = (np.random.rand(K) + 0.5) * 20
            P_tot = np.random.choice([200, 300, 500])

            t1 = time.perf_counter()
            f, p = optimize_p_kkt(D0, D1, E1, E2, G0, G1, G2, P_tot)
            print(f" kkt time: {time.perf_counter() - t1}")
            # assertions for kkt
            self.assertAlmostEqual(np.sum(p), P_tot, places=4)

            t1 = time.perf_counter()
            f1, p1 = optimize_p_quasisubgradient(D0, D1, E1, E2, G0, G1, G2, P_tot)
            print(f" QPA time: {time.perf_counter() - t1}")
            self.assertAlmostEqual(np.sum(p1), P_tot, places=4)
            # assertions for QPA
            self.assertGreaterEqual(f, f1)
            print(f"f = {f}, f1 = {f1}, f - f1 = {f - f1}, r_gap = {(f - f1)/f}")


if __name__ == '__main__':
    unittest.main()
