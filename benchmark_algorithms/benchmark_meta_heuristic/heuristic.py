# Solve the NOMA problem by meta-heuristic algorithms, such as GBO, SHIO, etc.

import numpy as np
from mealpy import PSO, SHIO, FloatVar
from mealpy.math_based import GBO
from mealpy.utils.problem import Problem
from scenarios.scenario_creators import create_scenario
from main_algorithms.bcd_algorithm import bcd
from benchmark_algorithms.benchmark_OMA.bcd_OMA import bcd_OMA
import math
from scenarios.scenario_creators import load_scenario


def violate_unequal(value):
    return 0 if value <= 0 else value

def violate_equal(value, eps=1e-4):
    return 0 if abs(value) <= eps else abs(value)

class NomaProblem(Problem):
    def __init__(self, sc, bounds, minmax, **kwargs):
        self.sc = sc
        self.K = int(len(self.sc.UEs)/2)  # K is the number of near users.
        super().__init__(bounds, minmax, **kwargs)

    # Let K be the number of near user. Then x[:K] are the bandwidth variables;
    # x[K:3*K] are the power variables: x[K:2*K] are the power allocated to near users, x[2*K: 3*K] are the power
    # allocated to far users.
    # x[3*K] is the squared height of the UAV.
    def con_x(self, x):
        """
        :param x: x should be a numpy array with length 3*K+2
        :return:
        """
        return np.sum(x[:self.K]) - self.sc.pn.b_tot


    def con_p(self, x):
        return np.sum(x[self.K: 3*self.K]) - self.sc.pn.p_max


    def con_h_z(self, x):
        uav = self.sc.uav
        R_u = self.sc.pn.radius + math.sqrt(uav.u_x**2 + uav.u_y**2)
        theta = max(self.sc.pn.theta_min ** 2, (math.atan(R_u / (np.sqrt(x[3 * self.K])))) ** 2)
        return R_u - math.sqrt(x[3*self.K]) * math.tan(math.sqrt(theta))



    def obj_func(self, solution):
        """
        The objective function value for mealpy.
        :param solution: numpy array, length = 3*K+1
        :return:
        """
        obj, K = 1e10, self.K
        pn, uav, UEs = self.sc.pn, self.sc.uav, self.sc.UEs
        R_u = self.sc.pn.radius + math.sqrt(uav.u_x ** 2 + uav.u_y ** 2)
        theta = max(pn.theta_min ** 2, (math.atan(R_u / (np.sqrt(solution[3 * self.K])))) ** 2)
        for k in range(K):
            rate_near = solution[k] * np.log1p(pn.g_0 * UEs[k].tilde_g * solution[k+K] / (
                    solution[k] * pn.sigma * theta * (solution[3*K] + (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2) ** (pn.alpha / 2)))

            rate_far_1 = solution[k] * np.log1p(pn.g_0 * UEs[k+K].tilde_g * solution[k+2*K] / (
                solution[k] * pn.sigma * theta * (solution[3*K] + (UEs[k+K].loc_x - uav.u_x) ** 2 + (UEs[k+K].loc_y - uav.u_y) ** 2) ** (pn.alpha/2)
                + pn.g_0 * UEs[k+K].tilde_g * solution[k+K]
            ))
            rate_far_2 = solution[k] * np.log1p(pn.g_0 * UEs[k].tilde_g * solution[k+2*K] / (
                solution[k] * pn.sigma * theta * (solution[3*K] + (UEs[k].loc_x - uav.u_x) ** 2 + (UEs[k].loc_y - uav.u_y) ** 2) ** (pn.alpha/2)
                + pn.g_0 * UEs[k].tilde_g * solution[k+K]
            ))
            obj = min(obj, rate_near, rate_far_1, rate_far_2)
        # penalty of constraint violation
        penalty_coefficient = 10000
        penalty = (penalty_coefficient * violate_unequal(self.con_x(solution)) + penalty_coefficient * violate_unequal(self.con_p(solution))
                   + penalty_coefficient * violate_unequal(self.con_h_z(solution)))
        # print(f"-obj = {-obj: .6f}, pl = {penalty: .6f}, p_x = {(self.con_x(solution)): .6f}, p_p = {(self.con_p(solution)): .4f},"
        #       f"p_h = {violate_unequal(self.con_h_z(solution)): .6f}")
        obj = -obj + penalty
        return obj

def get_lb_ub_BCD_based(sc):
    """
    Set the upper bound for meta-heuristic algorithm. ub and lb are set to values based on the optima of BCD.
    :param sc: the scenario
    :return: x_ub = sc.UEs[k].x*1.1, p_ub = sc.UEs[k].p*1.1
    """
    K = int(len(sc.UEs) / 2)
    R_u = sc.pn.radius + math.sqrt(sc.uav.u_x ** 2 + sc.uav.u_y ** 2)
    lb, ub = np.ones(3*K + 1)*1e-8, np.ones(3*K + 1)
    for k in range(K):
        ub[k] = sc.UEs[k].x * 1.1
        ub[k+K] = sc.UEs[k].p * 1.1
        ub[k+2*K] = sc.UEs[k+K].p * 1.1
    lb[-1] = max(sc.pn.h_min**2, (R_u / math.tan(sc.pn.theta_max)) ** 2)
    ub[-1] = min(sc.pn.h_max ** 2, (R_u / math.tan(sc.pn.theta_min)) ** 2)
    return lb, ub

def get_lb_ub(sc):
    """
    Set the upper bound for meta-heuristic algorithm. ub and lb are set with no other information.
    :param sc: the scenario
    :return: x_ub = sc.UEs[k].x*1.1, p_ub = sc.UEs[k].p*1.1
    """
    K = int(len(sc.UEs) / 2)
    R_u = sc.pn.radius + math.sqrt(sc.uav.u_x ** 2 + sc.uav.u_y ** 2)
    lb, ub = np.ones(3*K + 1)*1e-8, np.ones(3*K + 1)
    ub[:K] = sc.pn.b_tot*2/K
    ub[K:3*K] = sc.pn.p_max*2/K
    lb[-1] = max(sc.pn.h_min**2, (R_u / math.tan(sc.pn.theta_max)) ** 2)
    ub[-1] = min(sc.pn.h_max ** 2, (R_u / math.tan(sc.pn.theta_min)) ** 2)
    return lb, ub

def NOMA_heuristic(sc, method='GBO'):
    K = int(len(sc.UEs)/2)
    # solve the NOMA problem by meta-heuristics
    lb, ub = get_lb_ub_BCD_based(sc)
    # lb, ub = get_lb_ub(sc)

    prob = NomaProblem(sc, FloatVar(lb=lb, ub=ub), minmax="min")
    model = None
    if method == 'GBO':
        model = GBO.OriginalGBO(epoch=1000, pop_size=100)
    else:
        model = SHIO.OriginalSHIO(epoch=1000, pop_size=100)
    model.solve(prob)
    # model_shio = SHIO.OriginalSHIO(epoch=1000, pop_size=100)
    # model_shio.solve(prob)
    f_shio = -model.g_best.target.fitness

    return f_shio, model.g_best.solution


if __name__ == '__main__':
    sc = create_scenario(10, 100)

    sc.reset_scenario()
    sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()
    f_NOMA = bcd(sc, debug=False, tol=1e-6)
    x_NOMA = np.concatenate((np.array([u.x for u in sc.UEs]), np.array([u.p for u in sc.UEs]), np.array([sc.uav.h, sc.uav.theta])))
    print(f"f_NOMA = {f_NOMA}, x_NOMA = {x_NOMA}, rates = {sc.get_UE_rates()}")

    f_heu, x_heu = NOMA_heuristic(sc)
    K = int(len(sc.UEs)/2)
    sc.set_x(x_heu[:K])
    sc.set_p(x_heu[K:3*K])
    sc.uav.h = x_heu[3*K]

    R_u = sc.pn.radius + math.sqrt(sc.uav.u_x ** 2 + sc.uav.u_y ** 2)
    sc.uav.theta = max(sc.pn.theta_min ** 2, (math.atan(R_u / (np.sqrt(x_heu[3 * K])))) ** 2)

    rates = sc.get_UE_rates()
    print(f"f_heu = {f_heu: .6f}, rates = {rates}")

    sc.reset_scenario_OMA()
    sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()
    f_OMA = bcd_OMA(sc, debug=False, tol=1e-6)

    print(f"f_NOMA = {f_NOMA}, f_OMA = {f_OMA}, f_heu = {f_heu}, r_gap = {abs(f_heu - f_NOMA)/f_NOMA}")
    print(f"x_NOMA = {x_NOMA}, x_heu = {x_heu}")



    
