import pandas as pd
from matplotlib import pyplot as plt


def plot_convergence():
    fontsize = 20
    df = pd.read_excel('Bayesian_convergence.xlsx', index_col=0)
    plt.rcParams.update({'font.size': fontsize})
    df.plot(legend=False, lw=2, xlabel='Iterations', ylabel='Best objective value',
                 fontsize=fontsize, style=["b-o"], markersize=10, grid=True,
                 markerfacecolor='none')
    plt.xlim((-1, 21))
    plt.show()


if __name__ == '__main__':
    plot_convergence()
