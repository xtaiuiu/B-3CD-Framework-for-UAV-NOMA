import numpy as np
from scipy.stats import ncx2
import matplotlib.pyplot as plt

def rician_channel_gain(n=1, K_R=15.85):
    """
    generate the small scale rician fading coefficient, i.e., tilde_g in the article
    :return: the generated random number
    """
     # represent K_R = 12 dB
    coeff = 2*K_R + 2
    df, nonc = 2, 2*K_R
    rng = np.random.default_rng()
    if n == 1:
        return rng.noncentral_chisquare(df, nonc)/coeff
    else:
        return rng.noncentral_chisquare(df, nonc, n)/coeff


if __name__ == '__main__':
    values = np.array([rician_channel_gain() for _ in range(int(1e5))])
    values2 = np.array([rician_channel_gain(K_R=0) for _ in range(int(1e5))])
    plt.hist(values, bins=200, density=True, color='blue')
    plt.hist(values2, bins=200, density=True, color='red')
    plt.show()
