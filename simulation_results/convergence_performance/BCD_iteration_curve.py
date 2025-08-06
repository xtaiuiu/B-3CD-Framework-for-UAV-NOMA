# Plot the convergence curve of BCD algorithm
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_algorithms.bcd_algorithm import bcd


def run():
    project_root = Path(__file__).parent.parent.parent
    
    df = pd.DataFrame()
    
    # Process scenarios with 50, 100, and 150 UEs
    for i in [1, 2, 3]:
        num_ues = i * 50
        # Path to the pickle file
        pickle_path = project_root / 'main_algorithms' / f'scenario_{num_ues}_UEs.pickle'

        # Load the scenario
        with open(pickle_path, 'rb') as f:
            sc = pickle.load(f)
            sc.reset_scenario()

        # Run BCD algorithm and store results
        f_values = bcd(sc, tol=1e-4, debug=True)
        df[f'{num_ues} users'] = np.array(f_values)
    print(df)
    df.to_excel('BCD_iteration_curve.xlsx')


def plot():
    fontsize = 20
    step = 20

    df = pd.read_excel('BCD_iteration_curve.xlsx', index_col=0)[:21]
    G = np.arange(df.shape[0])
    plt.rcParams.update({'font.size': fontsize})
    ax = df.plot(legend=True, lw=2, xlabel='Number of iterations', ylabel='Function value',
                              fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                              markerfacecolor='none')
    # ax.set_xticks(G)
    plt.ylim((-1, 8.1))
    plt.xlim((-1, 21))
    # ax.set_xticklabels([str((i + 1) * step) for i in G], fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    #run()
    plot()
