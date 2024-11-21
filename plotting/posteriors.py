import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])


def plot_posterior(log_probs, samples, param):
    """
    log_probs: log of the probabilities
    samples: samples from the MCMC
    param: parameter to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    xy = np.vstack([samples[:, param], log_probs]).T
    xy = xy[xy[:, 0].argsort()]
    max_logprob = np.max(log_probs)
    xy = xy[-2*(xy[:, 1] - max_logprob) < 14]
    xy = np.maximum.reduceat(xy, np.arange(0, len(xy), 25))
    ax.plot(xy[:, 0], -2*(xy[:, 1] - max_logprob))

    ax.set_xlabel(f"Parameter {param}")
    ax.set_ylabel(r"$\Delta\chi^2_{\mathrm{min}}$")

    plt.plot()
    plt.show()