import numpy as np
import matplotlib.pyplot as plt


def nse(targets: np.ndarray, predictions: np.ndarray):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))

def normalize(x):
    return 1/(2 - x)

def evaluate(P: np.ndarray, E: np.ndarray, Q: np.ndarray, Q_hat:np.ndarray):

    # Calculate NSE score
    nse_score = nse(Q, Q_hat)
    nnse_score = normalize(nse_score)

    print(f"NSE: {nse_score:.3f}")
    print(f"Normalized NSE: {nnse_score:.3f}")


    # Plot hydrograph
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(Q, color='black', label='obs', alpha=1.0)
    ax.plot(Q_hat, color='red', label='pred', alpha=0.7)
    ax.plot(P, 'g--', label='precip', alpha=0.45)
    ax.plot(E, 'y--', label='etp', alpha=0.45)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Flow')

    plt.legend()

    return fig