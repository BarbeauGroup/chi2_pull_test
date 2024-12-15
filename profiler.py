from tqdm import tqdm
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

from flux.probabilities import sin2theta

import matplotlib.pyplot as plt

def utosin(u):
    return 4 * (u - u*u)

def sintou(sin):
    return 0.5 * (1 - np.sqrt(1 - sin))

def plot(filestems, labels, levels):
    colors = ["#2c7bb6", "#008837", "#e66101", "#7b3294"]
    colors = colors[:len(filestems)]

    sin2_bins = np.logspace(-6, 0, 200, endpoint=True)

    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()

    for file, label, color in zip(filestems, labels, colors):
        bins = np.load(file + "_bins_mass_uu.npz")
        chi2 = np.load(file + "_chi2_mass_uu.npy")
        params = np.load(file + "_params_mass_uu.npy")
        success = np.load(file + "_success_mass_uu.npy")
        u_bins = bins["u_bins"]
        mass_bins = bins["mass_bins"]
        sinx, siny = np.meshgrid(sin2_bins, mass_bins)
        ux, uy = np.meshgrid(u_bins, mass_bins)

        us4_chi2 = np.full((len(u_bins), len(mass_bins)), 1e6, dtype=float)
        sin2_ee_chi2 = np.full((len(sin2_bins), len(mass_bins)), 1e6, dtype=float)
        sin2_mm_chi2 = np.full((len(sin2_bins), len(mass_bins)), 1e6, dtype=float)
        sin2_em_chi2 = np.full((len(sin2_bins), len(mass_bins)), 1e6, dtype=float)

        ue4_mm = np.full((len(sin2_bins), len(mass_bins)), 2.0, dtype=float)
        us4_em = np.full((len(sin2_bins), len(mass_bins)), 2.0, dtype=float)
        umu4_ee = np.full((len(sin2_bins), len(mass_bins)), 2.0, dtype=float)

        ue4s = np.full((len(sin2_bins), len(mass_bins)), 2.0, dtype=float)
        umu4s = np.full((len(sin2_bins), len(mass_bins)), 2.0, dtype=float)

        for i, ue4 in enumerate(u_bins):
            for j, umu4 in enumerate(u_bins):
                for k, mass in enumerate(mass_bins):
                    if ue4 + umu4 > 1: continue
                    s2_ee = sin2theta(1, 1, ue4, 0, 0)
                    s2_mm = sin2theta(2, 2, 0, umu4, 0)
                    s2_em = sin2theta(1, 2, ue4, umu4, 0)
                    us4 = 1 - ue4 - umu4

                    idx_ee = np.searchsorted(sin2_bins, s2_ee)
                    idx_mm = np.searchsorted(sin2_bins, s2_mm)
                    idx_em = np.searchsorted(sin2_bins, s2_em)
                    idx_us4 = np.searchsorted(u_bins, us4)

                    if chi2[i, j, k] < us4_chi2[idx_us4, k]:
                        us4_chi2[idx_us4, k] = chi2[i, j, k]
                    if chi2[i, j, k] < sin2_ee_chi2[idx_ee, k]:
                        sin2_ee_chi2[idx_ee, k] = chi2[i, j, k]
                        umu4_ee[idx_ee, k] = umu4
                        ue4s[idx_ee, k] = ue4
                        umu4s[idx_ee, k] = umu4
                    if chi2[i, j, k] < sin2_mm_chi2[idx_mm, k]:
                        sin2_mm_chi2[idx_mm, k] = chi2[i, j, k]
                        ue4_mm[idx_mm, k] = ue4
                    if chi2[i, j, k] < sin2_em_chi2[idx_em, k]:
                        sin2_em_chi2[idx_em, k] = chi2[i, j, k]
                        us4_em[idx_em, k] = us4
                    
        # idx_s = np.searchsorted(sin2_bins, 0.99)
        # idx_m = np.searchsorted(mass_bins, 3)

        idx_s, idx_m = np.unravel_index(np.argmax(np.ma.masked_where(sin2_ee_chi2 > 100, sin2_ee_chi2)), sin2_ee_chi2.shape)
        idx_ue4 = np.searchsorted(u_bins, ue4s[idx_s, idx_m])
        idx_umu4 = np.searchsorted(u_bins, umu4s[idx_s, idx_m])
        idx_better_umu4 = np.searchsorted(u_bins, 1 - ue4s[idx_s, idx_m])

        # print(ue4s, umu4s)

        print(file)
        print("ue4", ue4s[idx_s, idx_m])
        print("umu4", umu4s[idx_s, idx_m])
        print("sin2", sin2_bins[idx_s])
        print("mass", mass_bins[idx_m])
        print("params", params[idx_ue4, idx_umu4, idx_m])
        print("params better umu4", params[idx_ue4, idx_better_umu4, idx_m])
        print("chi2", chi2[idx_ue4, idx_umu4, idx_m])
        print("chi2 better umu4", chi2[idx_ue4, idx_better_umu4, idx_m])
        print("success", success[idx_ue4, idx_umu4, idx_m])
        print("success better umu4", success[idx_ue4, idx_better_umu4, idx_m])

        
        # print(file)
        # print("ue4_mm", ue4_mm)
        # print("us4_em", us4_em)
        # print("umu4_ee", umu4_ee)

        us4_chi2_ma = np.ma.masked_where(us4_chi2 > 1e5, us4_chi2)
        sin2_ee_chi2_ma = np.ma.masked_where(sin2_ee_chi2 > 1e5, sin2_ee_chi2)
        sin2_mm_chi2_ma = np.ma.masked_where(sin2_mm_chi2 > 1e5, sin2_mm_chi2)
        sin2_em_chi2_ma = np.ma.masked_where(sin2_em_chi2 > 1e5, sin2_em_chi2)

        ax[0].contour(ux, uy, us4_chi2_ma.T, levels=levels, colors=color)
        ax[1].contour(sinx, siny, sin2_ee_chi2_ma.T, levels=levels, colors=color)
        ax[2].contour(sinx, siny, sin2_mm_chi2_ma.T, levels=levels, colors=color)
        ax[3].contour(sinx, siny, sin2_em_chi2_ma.T, levels=levels, colors=color)

        ax[3].plot([], [], color=color, label=label)
    
    # Plot BEST data on s_ee plot
    BEST_data1 = np.loadtxt("data/bad_data_thief/best_sage_gallex_pt1.csv", delimiter=",", skiprows=1)
    BEST_sin21 = BEST_data1[:, 0]
    BEST_mass1 = BEST_data1[:, 1]

    BEST_data2 = np.loadtxt("data/bad_data_thief/best_sage_gallex_pt2.csv", delimiter=",", skiprows=1)
    BEST_sin22 = BEST_data2[:, 0]
    BEST_mass2 = BEST_data2[:, 1]

    BEST_sin22 = np.append(BEST_sin22, BEST_sin22[0])
    BEST_mass2 = np.append(BEST_mass2, BEST_mass2[0])
    ax[1].fill(BEST_sin21, BEST_mass1, 10, color="red", alpha=0.5)
    ax[1].fill(BEST_sin22, BEST_mass2, 10, color="red", alpha=0.5)
    ax[1].fill([], [], color="red", alpha=0.5, label="GALLEX+SAGE+BEST 90% CL")

    # Plot uboone and lsnd1 data on s_em plot
    uboone_data = np.loadtxt("data/bad_data_thief/uboone-limits.csv", delimiter=",")
    uboone_sin2 = uboone_data[:, 0]
    uboone_mass = uboone_data[:, 1]

    lsnd1_data = np.loadtxt("data/bad_data_thief/lsnd1.csv", delimiter=",")
    lsnd1_sin2 = lsnd1_data[:, 0]
    lsnd1_mass = lsnd1_data[:, 1]

    lsnd1_sin2 = np.append(lsnd1_sin2, lsnd1_sin2[-1])
    lsnd1_mass = np.append(lsnd1_mass, lsnd1_mass[-1])

    ax[3].plot(uboone_sin2, uboone_mass, "red", alpha=0.5, label="MicroBooNE 95% CL")
    ax[3].fill_between(lsnd1_sin2, lsnd1_mass, 50, color="blue", alpha=0.5, label="LSND 90% CL")

    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(1e-4, 1)
        a.set_ylim(0.1, 50)
        a.set_ylabel(r"$\Delta m^2_{41}$")
    
    ax[0].set_xlim(0.1,1)
    ax[0].set_xlabel(r"$|U_{s4}|^2$")
    ax[1].set_xlabel(r"$\sin^2 2\theta_{ee}$")
    ax[2].set_xlabel(r"$\sin^2 2\theta_{\mu\mu}$")
    ax[3].set_xlabel(r"$\sin^2 2\theta_{e\mu}$")

    secax = ax[1].secondary_xaxis('top', functions=(sintou, utosin))
    secax.set_xlabel(r"$|U_{e4}|^2$")

    fig.suptitle("95% CL Contours")

    plt.legend()
    plt.show()

if __name__ == "__main__":

    plot(["output/pb_glass3_2dssb", "output/combined", "output/csi_1t"], ["20ty Pb Glass @ 20, 30, 40m", "combined", "CsI 3 ty"], [5.99])

