import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import numpy as np
from scipy.ndimage.filters import gaussian_filter

plt.style.use(['science'])

def plot_all_posteriors(log_probs, samples, labels, bins=100, ylim=14):
    """
    log_probs: log of the probabilities
    samples: samples from the MCMC
    labels: labels for the parameters
    """
    n_params = samples.shape[1]
    fig, ax = plt.subplots(3, -(n_params // -3), figsize=(18, 10), sharey=True)
    ax = ax.flatten()

    max_logprob = np.max(log_probs)
    argmax = np.argmax(log_probs)
    for i in range(n_params):
        pll = np.zeros(bins)
        edges = np.linspace(np.min(samples[:, i]), np.max(samples[:, i]), bins)
        xs = np.digitize(samples[:, i], bins=edges)

        for j in range(bins-1):
            mask = (xs == j+1)
            if np.any(mask):
                pll[j] = -2*(np.max(log_probs[mask]) - max_logprob)
            else:
                pll[j] = np.nan

        edges = edges[pll < ylim]
        pll = pll[pll < ylim]
        ax[i].plot(edges[:-1], pll[:-1])
        ax[i].axhline(y=1, color="black", linestyle="--")
        ax[i].axhline(y=4, color="red", linestyle="--")
        ax[i].axhline(y=9, color="blue", linestyle="--")

        ax[i].axvline(x=samples[argmax, i], color="black", linestyle="--")
        ax[i].text(samples[argmax, i] + 3*(edges[-1] - edges[0])/100, 12, f"{samples[argmax, i]:.2f}")

        ax[i].set_ylim(0, ylim+1)
        ax[i].set_xlabel(labels[i])
        if(i % 3 == 0):
            ax[i].set_ylabel(r"2$\Delta\ln L_{\mathrm{max}}$")
    
    for i in range(n_params, len(ax)):
        ax[i].axis("off")

    plt.tight_layout()
    plt.plot()
    plt.show()


def plot_posterior(log_probs, samples, xlabel, bins=100, ylim=14):
    """
    log_probs: log of the probabilities
    samples: samples from the MCMC
    param: parameter to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    max_logprob = np.max(log_probs)
    pll = np.zeros(bins)
    edges = np.linspace(np.min(samples), np.max(samples), bins)
    xs = np.digitize(samples, bins=edges)

    for i in range(bins-1):
        mask = (xs == i+1)
        if np.any(mask):
            pll[i] = -2*(np.max(log_probs[mask]) - max_logprob)
        else:
            pll[i] = np.nan

    pll = pll[pll < ylim]
    ax.plot(edges, pll)

    ax.set_xlabel(xlabel)
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
    
    xv, yv = np.meshgrid(x_edges[:-1], y_edges[:-1])
    
    # pll = gaussian_filter(pll, 0.9)
    ax.contour(xv, yv, pll[:-1, :-1].T, levels=[2.3, 6.18, 11.83], colors=["black", "red", "blue"], algorithm='serial')

    plt.plot()
    plt.show()