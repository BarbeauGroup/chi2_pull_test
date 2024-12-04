from tqdm import tqdm
import numpy as np

from flux.probabilities import sin2theta

import matplotlib.pyplot as plt

def plot_sin2theta(index):

    # load stuff to plot it with
    BEST_data1 = np.loadtxt("data/bad_data_thief/best_sage_gallex_pt1.csv", delimiter=",", skiprows=1)
    BEST_sin21 = BEST_data1[:, 0]
    BEST_mass1 = BEST_data1[:, 1]

    # BEST_sin21 = np.append(BEST_sin21, BEST_sin21[0])
    # BEST_mass1 = np.append(BEST_mass1, BEST_mass1[0])

    BEST_data2 = np.loadtxt("data/bad_data_thief/best_sage_gallex_pt2.csv", delimiter=",", skiprows=1)
    BEST_sin22 = BEST_data2[:, 0]
    BEST_mass2 = BEST_data2[:, 1]

    BEST_sin22 = np.append(BEST_sin22, BEST_sin22[0])
    BEST_mass2 = np.append(BEST_mass2, BEST_mass2[0])




    bins =  np.load("output/combined_bins_mass_uu.npz")# np.logspace(-2, 0, num=10), np.linspace(1, 2, 2)
    u_bins = bins["u_bins"]
    mass_bins = bins["mass_bins"]

    chi2_file = "output/combined_chi2_mass_uu.npy"
    success_file = "output/combined_success_mass_uu.npy"

    labels = ['90% CL']
    label = labels[0]

    # sin2_bins = np.linspace(0.01, 1, 50) 
    # ... kind of magic
    little_u_bins = np.linspace(0.01, 1, 100)
    sin2_bins = np.unique(sin2theta(1, 1, little_u_bins, 0, 0))[::2]
    sin2_bins = np.append(sin2_bins, 1)
    sin2_bins = np.unique(sin2_bins)
    
    sin2_chi2 = {label: np.full((len(sin2_bins), len(mass_bins)), 1e6) for label in labels}
    sin2_marginu = {label: np.zeros((len(sin2_bins), len(mass_bins))) for label in labels}

    chi2 = np.load(chi2_file)
    # margin_u = np.load(margin_file)
    suc = np.load(success_file)
    chi2 = np.ma.masked_where(suc == 0, chi2)
    # margin_u = np.ma.masked_where(suc == 0, margin_u)

    for i, ue4 in enumerate(u_bins):
        for j, umu4 in enumerate(u_bins):
            for k, mass in enumerate(mass_bins):
                if index == 1:
                    s2 = sin2theta(1, 1, ue4, 0, 0)
                elif index == 2:
                    s2 = sin2theta(2, 2, 0, umu4, 0)
                idx = np.searchsorted(sin2_bins, s2)
                if idx >= len(sin2_bins):
                    print(s2, idx)
                if chi2[i, j, k] < sin2_chi2[label][idx, k]:
                    sin2_chi2[label][idx, k] = chi2[i, j, k]

    # plot chi2
    fig, ax = plt.subplots()
    xv, yv = np.meshgrid(mass_bins, sin2_bins)

    sin2_chi2_ma = np.ma.masked_where(sin2_chi2[label] > 1e5, sin2_chi2[label])

    levels = [4.61]
    # ax.imshow(sin2_chi2_ma.T, extent=(sin2_bins.min(), sin2_bins.max(), mass_bins.min(), mass_bins.max()), aspect="auto", origin="lower", vmax=4.61)
    ax.contour(xv, yv, sin2_chi2_ma, levels=levels)

    # for i, x in enumerate(xv):
    #     for j, y in enumerate(yv):
    #         print(xv[i, j], yv[i, j], sin2_chi2_ma[i, j])

    ax.plot([], [], color="black", label=r"20t Pb glass, two baslines 95\% CL")

    ax.set_yscale("log")
    # ax.set_yscale("log")
    if index == 1:
        ax.set_ylabel(r"$\sin^2 2\theta_{ee}$")
    elif index == 2:
        ax.set_ylabel(r"$\sin^2 2\theta_{\mu\mu}$")
    ax.set_xlabel(r"$\Delta m^2_{41}$")
    ax.set_ylim(1e-2, 1)
    ax.set_xlim(0, 15)


    # add intervals for best
    ax.fill(BEST_sin21, BEST_mass1, 10, color="red", alpha=0.5)#, label="GALLEX+SAGE+BEST 90\% CL")
    ax.fill(BEST_sin22, BEST_mass2, 10, color="red", alpha=0.5)

    plt.legend()

    plt.show()

def plot_sin2theta_mue():

    # load stuff to plot it with

    uboone_data = np.loadtxt("data/bad_data_thief/uboone-limits.csv", delimiter=",")
    uboone_sin2 = uboone_data[:, 0]
    uboone_mass = uboone_data[:, 1]

    lsnd1_data = np.loadtxt("data/bad_data_thief/lsnd1.csv", delimiter=",")
    lsnd1_sin2 = lsnd1_data[:, 0]
    lsnd1_mass = lsnd1_data[:, 1]

    lsnd1_sin2 = np.append(lsnd1_sin2, lsnd1_sin2[-1])
    lsnd1_mass = np.append(lsnd1_mass, lsnd1_mass[-1])


    # 

    bins =  np.load("output/combined_bins_mass_uu.npz")# np.logspace(-2, 0, num=10), np.linspace(1, 2, 2)
    u_bins = bins["u_bins"]
    mass_bins = bins["mass_bins"]

    sin2_bins = np.logspace(-5, 0, 200, endpoint=True)
    # print(sin2_bins)
    # return
    # sin2_bins = np.unique(sin2theta(1, 2, little_u_bins, little_u_bins, 0))[::2]
    # sin2_bins = np.append(sin2_bins, 1)
    # sin2_bins = np.unique(sin2_bins)

    fig, ax = plt.subplots()

    xv, yv = np.meshgrid(sin2_bins, mass_bins)

    for chi2_file, color in zip(["output/csi_1t_chi2_mass_uu.npy", "output/combined_chi2_mass_uu.npy", "output/pb_glass3_chi2_mass_uu.npy"], ["black", "blue", "green"]):
    # chi2_file = "output/combined_chi2_mass_uu.npy"
        chi2 = np.load(chi2_file)

        sin2_chi2 = np.full((len(sin2_bins), len(mass_bins)), 1e6)

        for i, u1 in enumerate(u_bins):
            for j, u2 in enumerate(u_bins):
                for k, mass in enumerate(mass_bins):
                    if u1 + u2 > 1: continue
                    s2 = sin2theta(1, 2, u1, u2, 0)

                    idx = np.searchsorted(sin2_bins, s2)

                    if chi2[i, j, k] < sin2_chi2[idx, k]:
                        sin2_chi2[idx, k] = chi2[i, j, k]

        sin2_chi2_ma = np.ma.masked_where(sin2_chi2 > 1e5, sin2_chi2)
        # for i, s in enumerate(sin2_bins):
        #     for j, m in enumerate(mass_bins):
        #         print(s, m, sin2_chi2_ma[i, j])

        levels = [3,5,7,9,11]
        # ax.contour(xv, yv, sin2_chi2_ma.T, levels=levels, colors=["blue", "red", "purple", "black", "yellow"])
        ax.imshow(sin2_chi2_ma.T, origin='lower', aspect='auto', vmax=11)
        break

        # add a patch with this color so we can label it as two baseline pb glass 95% CL
        ax.plot([], [], color=color, label=chi2_file.split("/")[1].split("_")[0])

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # ax.set_xlabel(r"$\sin^2 2\theta_{\mu e}$")
    # ax.set_ylabel(r"$\Delta m^2_{41}$")
    # ax.set_xlim(sin2_bins.min(), sin2_bins.max())
    # ax.set_ylim(mass_bins.min(), mass_bins.max())
    # ax.set_xlim(5e-6, 1)
    # ax.set_ylim(0.1, 50)

    # add uboone limits
    # ax.plot(uboone_sin2, uboone_mass, "red", alpha=0.5, label=r"MicroBooNE 95\% CL")

    # lsnd1 is a closed interval so fill it
    # ax.fill_between(lsnd1_sin2, lsnd1_mass, 50, color="blue", alpha=0.5, label=r"LSND 90\% CL")

    plt.legend()

    plt.show()

if __name__ == "__main__":


    # plot_sin2theta(1)

    # plot_sin2theta(2)

    plot_sin2theta_mue()

