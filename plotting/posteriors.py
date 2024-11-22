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
    xy = np.maximum.reduceat(xy, np.arange(0, len(xy), 100))
    ax.plot(xy[:, 0], -2*(xy[:, 1] - max_logprob))

    ax.set_xlabel(f"Parameter {param}")
    ax.set_ylabel(r"$\Delta\chi^2_{\mathrm{min}}$")

    plt.plot()
    plt.show()

def plot_2dposterior(log_probs, samples, param1, param2, xbins=100, ybins=100):
    """
    log_probs: log of the probabilities
    samples: samples from the MCMC
    param1: parameter to plot
    param2: parameter to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    max_logprob = np.max(log_probs)
    
    pll = np.zeros((xbins, ybins))
    x_edges = np.linspace(np.min(samples[:, param1]), np.max(samples[:, param1]), xbins)
    y_edges = np.linspace(np.min(samples[:, param2]), np.max(samples[:, param2]), ybins)
    xs = np.digitize(samples[:, param1], bins=x_edges)
    ys = np.digitize(samples[:, param2], bins=y_edges)

    for i in range(xbins-1):
        for j in range(ybins-1):
            mask = (xs == i+1) & (ys == j+1)
            if(np.any(mask)):
                pll[i, j] = -2*(np.max(log_probs[mask]) - max_logprob)
            else:
                pll[i, j] = np.nan
    
    xv, yv = np.meshgrid(x_edges, y_edges)

    ax.contour(xv, yv, pll, levels=[2.3, 6.18, 11.83], colors=["black", "blue", "red"])

    plt.plot()
    plt.show()