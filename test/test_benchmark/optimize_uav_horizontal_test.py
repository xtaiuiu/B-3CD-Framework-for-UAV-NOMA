import unittest

import numpy as np

from benchmark_algorithms.benchmark_OMA.bcd_OMA import bcd_OMA
from benchmark_algorithms.benchmark_OMA.horizontal_Bayesian_OMA import optimize_horizontal_Bayesian_OMA
from main_algorithms.bcd_algorithm import bcd
from main_algorithms.optimize_uav_position_Bayesian import optimize_horizontal_Bayesian
from scenarios.scenario_creators import create_scenario
from simulation_results.horizontal_performance.user_distributions import road_based_distribution, \
    create_ue_set_with_distribution, asymmetric_gaussian_distribution


class MyTestCase(unittest.TestCase):
    def test_optimize_horizontal_Bayesian(self):
        # generate asymmetric user distribution
        n_ues = np.random.randint(50, 201)
        sc = create_scenario(n_ues, 100)
        UE_set_asymmetric = create_ue_set_with_distribution(round(n_ues / 6) * 6, 100, 0.5, 10,
                                                            asymmetric_gaussian_distribution, plot=False)
        sc.UEs = UE_set_asymmetric
        sc.reset_scenario()
        sc.plot_scenario_kde()
        sc.uav.u_x, sc.uav.u_y = 0, 0  # cell center
        f_cell_center = bcd(sc)

        sc.reset_scenario()
        sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()  # user center
        f_user_center = bcd(sc)

        sc.reset_scenario()
        sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()  # near center
        f_near_center = bcd(sc)

        sc.reset_scenario()
        f_Bayesian, _ = optimize_horizontal_Bayesian(sc)  # Bayesian
        print(f"NOMA: f_cell_center = {f_cell_center}, f_user_center={f_user_center}, f_near_center={f_near_center}, f_Bayesian={f_Bayesian}")

        self.assertGreaterEqual(f_Bayesian, f_cell_center)
        self.assertGreaterEqual(f_Bayesian, f_user_center)
        self.assertGreaterEqual(f_Bayesian, f_near_center)

    def test_optimize_horizontal_Bayesian_OMA(self):
        # generate asymmetric user distribution
        n_ues = np.random.randint(50, 201)
        sc = create_scenario(n_ues, 100)
        UE_set_asymmetric = create_ue_set_with_distribution(round(n_ues / 6) * 6, 100, 0.5, 10,
                                                            asymmetric_gaussian_distribution, plot=False)
        sc.UEs = UE_set_asymmetric
        sc.reset_scenario_OMA()
        sc.plot_scenario_kde()
        sc.uav.u_x, sc.uav.u_y = 0, 0  # cell center
        f_cell_center = bcd_OMA(sc)

        sc.reset_scenario_OMA()
        sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()  # user center
        f_user_center = bcd_OMA(sc)

        sc.reset_scenario_OMA()
        sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()  # near center
        f_near_center = bcd_OMA(sc)

        sc.reset_scenario_OMA()
        f_Bayesian, _ = optimize_horizontal_Bayesian_OMA(sc)  # Bayesian

        print(
            f"OMA: f_cell_center = {f_cell_center}, f_user_center={f_user_center}, f_near_center={f_near_center}, f_Bayesian={f_Bayesian}")

        self.assertGreaterEqual(f_Bayesian, f_cell_center)
        self.assertGreaterEqual(f_Bayesian, f_user_center)
        self.assertGreaterEqual(f_Bayesian, f_near_center)


if __name__ == '__main__':
    unittest.main()
