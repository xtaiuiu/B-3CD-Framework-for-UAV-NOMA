import time

import numpy as np
from GPyOpt.methods import BayesianOptimization
from matplotlib import pyplot as plt
from matplotlib.pyplot import grid, savefig
from pyinstrument import Profiler
import GPyOpt

from main_algorithms.bcd_algorithm import bcd
from scenarios.scenario_creators import load_scenario

class MyBayesianOptimization(BayesianOptimization):
    # an override BO class, which only alters the plotting functions
    def plot_convergence(self, filename=None):
        '''
        Plots to evaluate the convergence of standard Bayesian optimization algorithms
        '''
        Xdata, best_Y = self.X, self.Y_best
        n = Xdata.shape[0]
        plt.figure(figsize=(8, 6))
        # Estimated m(x) at the proposed sampling points
        plt.plot(list(range(n)), -best_Y, '-o')
        plt.title(r'Convergence of $B^3CD$')
        plt.xlabel('Iteration')
        plt.ylabel('Best Objective Value')
        grid(True)

        if filename != None:
            savefig(filename)
        else:
            plt.show()


def optimize_horizontal_Bayesian(sc, eps=1e-6, filename=None):
    def bcd_wrapper(X):  # a wrapper function for the bcd algorithm. As the GPyOpt library expects a function of the form f(x),
        # where X is an 2-dimensional array, and this library does not accept functions with extra arguments, we use a wrapper

        # Initialize output array with same number of rows as input X
        results = np.zeros((X.shape[0], 1))

        # Process each input point
        for i in range(X.shape[0]):
            x = X[i, 0]  # x-coordinate
            y = X[i, 1]  # y-coordinate
            # load the scenario
            sc.reset_scenario()
            sc.uav.u_x, sc.uav.u_y = x, y
            f = bcd(sc)
            results[i, 0] = -f  # store the negative value of the objective function

        return results  # return 2D array of results

    R = sc.pn.radius
    bounds = [{'name': 'x', 'type': 'continuous', 'domain': (-R, R)},
              {'name': 'y', 'type': 'continuous', 'domain': (-R, R)}]

    t = time.perf_counter()
    # Get center coordinates and create initial points array
    center_x, center_y = sc.get_UEs_center()
    center_near_x, center_near_y = sc.get_near_UEs_center()
    X = np.array([
        [0, 0],                         # Center of cell
        [center_x, center_y],           # Center of UEs
        [center_near_x, center_near_y], # Center of near UEs
        [R/2, R/2],                     # Top-right quadrant point
        [-R/2, R/2],                    # Top-left quadrant point
        [R/2, -R/2],                    # Bottom-right quadrant point
        [-R/2, -R/2],                    # Bottom-left quadrant point
        [R/4, R/4],                     # Top-right quarter point
        [-R/4, R/4],                    # Top-left quarter point
        [R/4, -R/4],                    # Bottom-right quarter point
        [-R/4, -R/4],                   # Bottom-left quarter point
    ])
    Y = bcd_wrapper(X)
    #print(Y)
    myProblem = MyBayesianOptimization(bcd_wrapper, bounds, X=X, Y=Y, normalize_Y=True, exact_feval=True)
    myProblem.run_optimization(max_iter=15, eps=1e-8, report_file='report.txt')
    myProblem.save_report('saved_report.txt')
    myProblem.save_evaluations("saved_evaluations.csv")
    # myProblem.plot_acquisition(label_x="x", label_y="y")
    myProblem.plot_convergence(filename=filename)
    plt.show()

    # print(f"runtime without X and Y in {time.perf_counter() - t} seconds")

    print(f"NOMA: f_opt = {-myProblem.fx_opt}, x_opt = {myProblem.x_opt}")
    return -myProblem.fx_opt, myProblem.x_opt



if __name__ == "__main__":
    sc = load_scenario('road_100_UEs.pickle')
    # sc.plot_scenario_kde()
    # sc.reset_scenario()
    #
    # sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()
    # f_center = bcd(sc)  # the objective function at the center of the points.
    #
    # sc.uav.u_x, sc.uav.u_y = 0, 0
    # f_cell_center = bcd(sc)  # the objective function at the center of the cell.

    f_Bayes, x_Bayes = optimize_horizontal_Bayesian(sc, filename='convergence_curve.png')
    # print(f"f_center = {f_center}, f_Bayes = {f_Bayes}, diff = {f_center - f_Bayes}")
    # print(f"f_cell_center = {f_cell_center}, f_Bayes = {f_Bayes}, diff = {f_cell_center - f_Bayes}")
    # print(f"x_center = {sc.get_UEs_center()}, x_Bayes = {x_Bayes}")