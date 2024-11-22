import numpy as np
import hdf5plugin
import h5py

import matplotlib.pyplot as plt
import scienceplots


def main():

    # var = 5
    hf = h5py.File('backend.h5', 'r')


    samples = hf['mcmc/chain'][:]
    log_prob = hf['mcmc/log_prob'][:]

    min_log_prob = np.min(log_prob)

    x = samples[:,:,2].flatten()
    delta_chi2 = -2*(log_prob.flatten())

    # Manually construct a histogram...
    bins = np.linspace(min(x), max(x), 50)
    min_chi2 = 1e4*np.ones(len(bins)-1)
    for i,ibin in enumerate(bins):
        if i == 0: continue
        for j in range(len(x)):
            if x[j] < bins[i] and x[j] > bins[i-1]:
                if delta_chi2[j] < min_chi2[i-1]:
                    min_chi2[i-1] = delta_chi2[j]


    min_min = np.min(min_chi2)
    
    plt.style.use('science')
    plt.scatter(x, delta_chi2 - min_min, s=1, alpha=0.1, label="All Samples")
    plt.plot(bins[:-1], min_chi2 - min_min, label="PLL", color='red')
    plt.ylabel(r"$\Delta \chi^2$")
    # plt.xscale('log')
    plt.axhline(1, color='black', linestyle='--', label=r"$1\sigma$")
    plt.xlabel(r"$|U_{\mu4}|^2$")
    plt.ylim(0,10)
    plt.legend()
    plt.show()



    return




    fig, axes = plt.subplots(8, figsize=(10, 7), sharex=True)
    samples = hf['mcmc/chain'][:]
    labels=[r"$\Delta m_{41}^2$", r"$|U_{e4}|^2$", r"$|U_{\mu 4}|^2$", r"$\alpha_{flux}$", r"$\alpha_{brn}$", r"$\alpha_{nin}$", r"$\alpha_{ssb}$", r"$\Delta t$"]
    for i in range(8):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.show()




if __name__ == "__main__":
    main()