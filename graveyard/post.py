import numpy as np
import hdf5plugin
import h5py

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scienceplots

from flux.probabilities import sin2theta


def main():

    # var = 5
    hf = h5py.File('backend_long.h5', 'r')


    samples = hf['mcmc/chain'][:]
    log_prob = hf['mcmc/log_prob'][:]


    chi2 = -2 * log_prob[:,].flatten()
    chi2 -= np.min(chi2)

    m = samples[:, :, 0].flatten()
    ue4 = samples[:, :, 1].flatten()
    um4 = samples[:, :, 2].flatten()

    sin_theta_ee = sin2theta(1, 1, ue4, um4, 0)
    sin_theta_me = sin2theta(1, 2, ue4, um4, 0)
    sin_theta_mm = sin2theta(2, 2, ue4, um4, 0)


    plot_me = False
    plot_ee = False
    plot_mm = True

    if plot_me:

        xbins = np.linspace(0.01, np.max(sin_theta_me), 100)
        ybins = np.linspace(0.01, np.max(m), 100)

        chi2_map = np.zeros((len(xbins)-1, len(ybins)-1))

        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):

                idx = np.where((m > ybins[i]) & (m < ybins[i+1]) & (sin_theta_me > xbins[j]) & (sin_theta_me < xbins[j+1]))[0]

                if len(idx) > 0:
                    chi2_map[i, j] = np.min(chi2[idx])
                else:
                    chi2_map[i, j] = 1e6

        min_idx = np.argmin(chi2)


        plt.style.use(['science'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.contourf(xbins[:-1], ybins[:-1], chi2_map, levels=[5.991, np.max(chi2_map)], colors=["gray"], alpha=0.5)
        ax.scatter(sin_theta_ee[min_idx], m[min_idx], color="red", label="Best Fit")

        ax.set_ylabel(r"$\Delta m^2_{41}$", fontsize=16)
        ax.set_xlabel(r"$\sin^2\theta_{\mu e}$", fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.01, np.max(sin_theta_me))
        ax.set_ylim(0.01, np.max(m))

        # Add the label with a box and color
        box = Patch(facecolor='gray', edgecolor='none', alpha=0.5, label=r"95\% CL")
        ax.legend(handles=[box], loc="upper left")
        plt.show()

    if plot_ee:

        xbins = np.linspace(0.01, np.max(sin_theta_ee), 100)
        ybins = np.linspace(0.01, np.max(m), 100)
            
        chi2_map = np.zeros((len(xbins)-1, len(ybins)-1))

        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):

                idx = np.where((m > ybins[i]) & (m < ybins[i+1]) & (sin_theta_ee > xbins[j]) & (sin_theta_ee < xbins[j+1]))[0]

                if len(idx) > 0:
                    chi2_map[i, j] = np.min(chi2[idx])
                else:
                    chi2_map[i, j] = 1e6

        min_idx = np.argmin(chi2)


        plt.style.use(['science'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # 
        ax.contourf(xbins[:-1], ybins[:-1], chi2_map, levels=[5.991, np.max(chi2_map)], colors=["gray"], alpha=0.5)
        ax.scatter(sin_theta_ee[min_idx], m[min_idx], color="red", label="Best Fit")

        ax.set_ylabel(r"$\Delta m^2_{41}$", fontsize=16)
        ax.set_xlabel(r"$\sin^2\theta_{ee}$", fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.01, np.max(sin_theta_ee))
        ax.set_ylim(0.01, np.max(m))

        # Add the label with a box and color
        box = Patch(facecolor='gray', edgecolor='none', alpha=0.5, label=r"95\% CL")
        ax.legend(handles=[box], loc="upper left")
        plt.show()

    if plot_mm:
        
        xbins = np.linspace(0.01, np.max(sin_theta_mm), 100)
        ybins = np.linspace(0.01, np.max(m), 100)

        chi2_map = np.zeros((len(xbins)-1, len(ybins)-1))

        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):

                idx = np.where((m > ybins[i]) & (m < ybins[i+1]) & (sin_theta_mm > xbins[j]) & (sin_theta_mm < xbins[j+1]))[0]

                if len(idx) > 0:
                    chi2_map[i, j] = np.min(chi2[idx])
                else:
                    chi2_map[i, j] = 1e6

        min_idx = np.argmin(chi2)

        plt.style.use(['science'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.contourf(xbins[:-1], ybins[:-1], chi2_map, levels=[5.991, np.max(chi2_map)], colors=["gray"], alpha=0.5)
        ax.scatter(sin_theta_mm[min_idx], m[min_idx], color="red", label="Best Fit")

        ax.set_ylabel(r"$\Delta m^2_{41}$", fontsize=16)
        ax.set_xlabel(r"$\sin^2\theta_{\mu\mu}$", fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.01, np.max(sin_theta_mm))
        ax.set_ylim(0.01, np.max(m))

        # Add the label with a box and color
        box = Patch(facecolor='gray', edgecolor='none', alpha=0.5, label=r"95\% CL")
        ax.legend(handles=[box], loc="upper left")

        plt.show()
        

            




if __name__ == "__main__":
    main()