import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from main_algorithms.bcd_algorithm import bcd
from scenarios.scenario_creators import create_scenario


def run(n_repeats=100):
    eps = np.array([1e-4, 1e-6, 1e-8])
    n_ues = np.arange(50, 251, 25)

    df_runtime_avg = pd.DataFrame({'1e-4': np.zeros(len(n_ues)), '1e-8': np.zeros(len(n_ues)),
                                   '1e-12': np.zeros(len(n_ues))})

    for repeat in range(n_repeats):
        t_r = time.perf_counter()
        print(f" ####################### repeat = {repeat} ##############################")
        df_runtime = pd.DataFrame({'1e-4': np.zeros(len(n_ues)), '1e-8': np.zeros(len(n_ues)),
                                   '1e-12': np.zeros(len(n_ues))})
        for i in range(len(n_ues)):
            print(f" ####################### repeat = {repeat}, N = {n_ues[i]} ##############################")
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            sc = create_scenario(n_ues[i], 100)
            sc.reset_scenario()
            for j in range(len(eps)):
                t = time.perf_counter()
                _ = bcd(sc, tol=eps[j])
                df_runtime.iloc[i, j] = time.perf_counter() - t
        print(f" ############## repeat = {repeat} in {time.perf_counter() - t_r} seconds ######################")
        df_runtime_avg += df_runtime
    df_runtime_avg /= n_repeats
    df_runtime_avg.to_excel('BCD_runtime_avg.xlsx')


def plot():
    fontsize = 20
    df = pd.read_excel('BCD_runtime_avg.xlsx', index_col=0)
    G = np.arange(df.shape[0])
    df.columns = [r'$\epsilon=10^{-4}$', r'$\epsilon=10^{-6}$', r'$\epsilon=10^{-8}$']
    plt.rcParams.update({'font.size': fontsize})
    ax = df.plot(legend=True, lw=2, xlabel='Number of users', ylabel='Convergence time (s)',
                 fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                 markerfacecolor='none')
    # Show only every other x-tick label
    x_ticks = [(i+2)*50 for i in G]
    ax.set_xticks(range(len(x_ticks)))  # Set ticks at all positions
    ax.set_xticklabels([str(x) if i % 2 == 0 else '' for i, x in enumerate(x_ticks)], fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    # run(n_repeats=10)
    plot()
