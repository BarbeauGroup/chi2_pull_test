import iminuit
from scipy.optimize import Bounds
from multiprocessing import Pool
from itertools import product
from utils import istarmap
from functools import partial

from tqdm import tqdm
import numpy as np
import os
from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt

from flux.probabilities import sin2theta, Pab

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from flux.ssb_pdf import make_ssb_pdf
from plotting.observables import analysis_bins, plot_observables, project_histograms, plot_observables2d
from plotting.posteriors import plot_posterior, plot_2dposterior
from stats.likelihood import loglike_stat, loglike_sys, loglike_stat_asimov

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from contourpy.util.bokeh_renderer import BokehRenderer as Renderer

# # Global variables for data
# params = load_params("config/csi.json")

# pkl = False
# new_time_edges = np.arange(0, 6625, 125)
# if not pkl or not os.path.exists("flux/flux_dict.pkl"):
#     flux = read_flux_from_root(params, new_time_edges)
#     with open("flux/flux_dict.pkl", "wb") as f:
#         np.save(f, flux)
# else:
#     with open("flux/flux_dict.pkl", "rb") as f:
#         flux = np.load(f, allow_pickle=True).item()
#     if(flux["nuE"][0][0] != new_time_edges).all():
#         flux = read_flux_from_root(params, new_time_edges)
#         with open("flux/flux_dict.pkl", "wb") as f:
#             np.save(f, flux)

# ssb_dict = make_ssb_pdf(params)
# bkd_dict = read_brns_nins_from_txt(params)
# data_dict = read_data_from_txt(params)

# observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
# t_bin_arr = np.asarray(params["analysis"]["time_bins"])
# flux_matrix = np.load(params["detector"]["flux_matrix"])
# detector_matrix = np.load(params["detector"]["detector_matrix"])
# matrix = detector_matrix @ flux_matrix

# u_bins = np.linspace(0.01, 1, 90)
# mass_bins = np.linspace(0, 30, 30)

def cost_asimov(x, mass, u, index, u2=None, /):
    """
    x is [time_offset, a_flux, a_brn, a_nin, a_ssb, other_u]
    index is 1 for Ue4, 2 for Umu4, 3 for Utau4
    """

    time_offset = x[0]

    osc_params = [params["detector"]["distance"]/100., mass, 0, 0, 0.0]
    if index == 1:
        osc_params[2] = u
        if u2 is not None:
            osc_params[3] = u2
        else:
            osc_params[3] = x[-1]
    elif index == 2:
        if u2 is not None:
            osc_params[2] = u2
        else:
            osc_params[2] = x[-1]
        osc_params[3] = u
    else:
        raise ValueError("Invalid index")

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)
    osc_obs = create_observables(params=params, flux=oscillated_flux, time_offset=0, matrix=matrix, flavorblind=True)
    hist_osc = analysis_bins(observable=osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=0)

    unosc_obs = create_observables(params=params, flux=flux, time_offset=time_offset, matrix=matrix, flavorblind=True)
    hist_unosc = analysis_bins(observable=unosc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=time_offset)

    nuisance_param_priors = [params["detector"]["systematics"]["flux"],
                            params["detector"]["systematics"]["brn"],
                            params["detector"]["systematics"]["nin"],
                            params["detector"]["systematics"]["ssb"]]

    return -2 * (loglike_stat_asimov(hist_osc, hist_unosc, x[1:-1], factor=10) + loglike_sys(x[1:-1], nuisance_param_priors))

def cost(x, mass, u, index, u2=None, /):
    """
    x is [time_offset, a_flux, a_brn, a_nin, a_ssb, other_u]
    index is 1 for Ue4, 2 for Umu4, 3 for Utau4
    """

    time_offset = x[0]

    osc_params = [params["detector"]["distance"]/100., mass, 0, 0, 0.0]
    if index == 1:
        osc_params[2] = u
        if u2 is not None:
            osc_params[3] = u2
        else:
            osc_params[3] = x[-1]
            x = x[:-1]
    elif index == 2:
        if u2 is not None:
            osc_params[2] = u2
        else:
            osc_params[2] = x[-1]
            x = x[:-1]
        osc_params[3] = u
    else:
        raise ValueError("Invalid index")

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)
    osc_obs = create_observables(params=params, flux=oscillated_flux, time_offset=time_offset, matrix=matrix, flavorblind=True)
    hist_osc = analysis_bins(observable=osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=time_offset)

    nuisance_param_priors = [params["detector"]["systematics"]["flux"],
                            params["detector"]["systematics"]["brn"],
                            params["detector"]["systematics"]["nin"],
                            params["detector"]["systematics"]["ssb"]]

    return -2 * (loglike_stat(hist_osc, x[1:]) + loglike_sys(x[1:], nuisance_param_priors))

def plot_u(index):
    chi2_files = ["output/chi2_csi_mass_ue4.npy"]
    margin_files = ["output/chi2_csi_mass_ue4_margin_umu4.npy"]
    success_files = ["output/chi2_csi_mass_ue4_success.npy"]
    colorlabels = ["white"]
    labels = ["90% CL"]


    # plot chi2
    fig, ax = plt.subplots()
    xv, yv = np.meshgrid(u_bins, mass_bins)

    for suc_file, margin_file, file, color, label in zip(success_files, margin_files, chi2_files, colorlabels, labels):
        suc = np.load(suc_file)
        chi2 = np.load(file)
        margin_u = np.load(margin_file)
        chi2 = np.ma.masked_where(suc == 0, chi2)
        margin_u = np.ma.masked_where(suc == 0, margin_u)
        # im = ax.imshow(margin_u.T, aspect='auto', origin='lower', extent=[u_bins.min(), u_bins.max(), mass_bins.min(), mass_bins.max()], cmap='viridis', norm=colors.LogNorm(vmin=0.01, vmax=1))
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(r'$|U_{\mu 4}|^2$')
        # ax.contour(xv, yv, chi2.T, levels=[4.61], algorithm='serial', colors=color, label=label)
        im = ax.imshow(chi2.T, aspect='auto', origin='lower', extent=[u_bins.min(), u_bins.max(), mass_bins.min(), mass_bins.max()], vmin=2.29, vmax=2.31, cmap='viridis',) # vmin=4.61, vmax=4.62
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'$\Delta\chi^2$')

    # set background yellow
    ax.set_facecolor('pink')

    # patches = [plt.Line2D([0], [0], color=color, label=label) for color, label in zip(colorlabels, labels)]
    # plt.legend(handles=patches)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if index == 1:
        ax.set_xlabel(r"$|U_{e 4}|^2$")
    elif index == 2:
        ax.set_xlabel(r"$|U_{\mu 4}|^2$")
    ax.set_ylabel(r"$\Delta m^2_{41}$")
    ax.set_xlim(0.01, 1)
    ax.set_ylim(10**-0.5, 50)

    plt.show()

def plot_sin2theta(index):
    u_bins, mass_bins =  np.linspace(0.01, 1, 100), np.linspace(0, 5, 10)
    # u_bins, mass_bins = np.linspace(0.01, 1, 1000), np.linspace(0, 50, 100)
    # chi2_files = ["chi2_10.npy", "chi2_50.npy", "chi2_100.npy", "chi2_1000.npy", "chi2_10*.npy", "chi2_10m.npy"]
    chi2_files = ["output/pbglass_chi2_mass_uu.npy"]
    # margin_files = ["output/chi2_csi_mass_ue4_margin_umu4.npy"]
    success_files = ["output/pbglass_success_mass_uu.npy"]
    # colors = ['blue', 'green', 'red', 'black', 'purple', 'orange']
    linecols = ['white']
    # labels = ['10', '50', '100', '1000', '10*', '10m']
    labels = ['90% CL']

    # sin2_bins = np.linspace(0.01, 1, 50)
    little_u_bins = np.linspace(0.01, 1, 100)
    sin2_bins = np.unique(sin2theta(1, 1, little_u_bins, 0, 0))[::3]
    sin2_bins = np.append(sin2_bins, 1)
    sin2_bins = np.unique(sin2_bins)
    print(sin2_bins)
    
    sin2_chi2 = {label: np.full((len(sin2_bins), len(mass_bins)), 1e6) for label in labels}
    sin2_marginu = {label: np.zeros((len(sin2_bins), len(mass_bins))) for label in labels}

    for success_file, chi2_file, label in zip(success_files, chi2_files, labels):
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
                    # elif index == 2:
                    #     s2 = sin2theta(2, 2, 0, umu4, 0)
                    idx = np.searchsorted(sin2_bins, s2)
                    if idx >= len(sin2_bins):
                        print(s2, idx)
                    if chi2[i, j, k] < sin2_chi2[label][idx, k]:
                        sin2_chi2[label][idx, k] = chi2[i, j, k]
                        # sin2_marginu[label][idx, j] = np.sum(Pab(np.arange(1, 60, 1), 19.3, mass, 1, 1, u, margin_u[i, j], 0))/59.
                        # sin2_marginu[label][idx, j] = margin_u[i, j]

    # plot chi2
    fig, ax = plt.subplots()
    xv, yv = np.meshgrid(sin2_bins, mass_bins)

    for color, label in zip(linecols, labels):
        # cont_gen = contour_generator(z=sin2_chi2_ma.T)
        # line = cont_gen.lines(4.61)
        # print(line)
        # renderer = Renderer(figsize=(8, 6))
        # renderer.lines(line, cont_gen.line_type, color="red", linewidth=2)
        # renderer.show()
        sin2_chi2_ma = np.ma.masked_where(sin2_chi2[label] > 1e5, sin2_chi2[label])
        im = ax.imshow(sin2_chi2_ma.T, aspect='auto', origin='lower', extent=[sin2_bins.min(), sin2_bins.max(), mass_bins.min(), mass_bins.max()], vmax=30, cmap='viridis',) # vmin=4.61, vmax=4.62
        # im = ax.imshow(sin2_chi2_ma.T, aspect='auto', origin='lower', extent=[sin2_bins.min(), sin2_bins.max(), mass_bins.min(), mass_bins.max()], vmin=6.17, vmax=6.19, cmap='Purples',) # vmin=4.61, vmax=4.62

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'$\Delta\chi^2$')
        # ax.contour(xv, yv, sin2_chi2_ma.T, levels=[0, 4.61, 5], algorithm='serial', colors=['white', 'white', 'white'],  linewidths=1)
        # ax.matshow(sin2_chi2_ma.T)

    # set background yellow
    # ax.set_facecolor('hotpink')

    # patches = [plt.Line2D([0], [0], color=color, label=label) for color, label in zip(linecols, labels)]
    # plt.legend(handles=patches)

    ax.set_xscale("log")
    # ax.set_yscale("log")
    if index == 1:
        ax.set_xlabel(r"$\sin^2 2\theta_{ee}$")
    elif index == 2:
        ax.set_xlabel(r"$\sin^2 2\theta_{\mu\mu}$")
    ax.set_ylabel(r"$\Delta m^2_{41}$")
    ax.set_xlim(1e-2, 1)
    ax.set_ylim(0.5, 5)

    # Add grid
    # ax.set_xticks(sin2_bins, minor=True)
    # ax.set_xticks(sin2_bins[::8])
    # ax.set_yticks(mass_bins, minor=True)
    # ax.set_yticks(mass_bins[::20])
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()

def plot_sin2theta_mue():
    u_bins, mass_bins = np.linspace(0.01, 1, 100), np.linspace(0, 5, 10)
    chi2 = np.load("output/pb_glass_chi2_mass_uu.npy")

    # sin2_bins = np.sort(np.unique(sin2theta(1, 2, np.tile(u_bins, len(u_bins)), np.repeat(u_bins, len(u_bins)), 0)))
    # sin2_bins = sin2_bins[np.where(sin2_bins < 1)]
    # sin2_bins = sin2_bins[::300]
    # sin2_bins = np.append(sin2_bins, 1)
    # print(len(sin2_bins))
    sin2_bins = np.linspace(0, 1, 100, endpoint=True)

    sin2_chi2 = np.full((len(sin2_bins), len(mass_bins)), 1e6)

    for i, u1 in enumerate(u_bins):
        for j, u2 in enumerate(u_bins):
            for k, mass in enumerate(mass_bins):
                s2 = sin2theta(1, 2, u1, u2, 0)

                idx = np.searchsorted(sin2_bins, s2)
                if chi2[i, j, k] < sin2_chi2[idx, k]:
                    sin2_chi2[idx, k] = chi2[i, j, k]

    fig, ax = plt.subplots()
    # sin2_chi2_ma = np.ma.masked_where(sin2_chi2 > 1e5, sin2_chi2)
    sin2_chi2_ma = np.ma.masked_where(sin2_chi2 > 1e5, sin2_chi2)
    im = ax.imshow(sin2_chi2_ma.T, aspect='auto', origin='lower', extent=[sin2_bins.min(), sin2_bins.max(), mass_bins.min(), mass_bins.max()], vmin=2.29, vmax=2.31, cmap='Greys',) # vmin=4.61, vmax=4.62
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$\Delta\chi^2$')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sin^2 2\theta_{e\mu}$")
    ax.set_ylabel(r"$\Delta m^2_{41}$")
    ax.set_xlim(sin2_bins.min(), sin2_bins.max())
    ax.set_ylim(mass_bins.min(), mass_bins.max())

    # # Add grid
    # ax.set_xticks(sin2_bins, minor=True)
    # ax.set_xticks(sin2_bins[::8])
    # ax.set_yticks(mass_bins, minor=True)
    # ax.set_yticks(mass_bins[::20])

    plt.show()

def marginalize_mass_u_helper(i, j, /, index):
    u = u_bins[i]
    mass = mass_bins[j]
    bounds = Bounds([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, 1-u], keep_feasible=True)
    res = iminuit.minimize(cost, np.asarray([80, 0, 0, 0, 0, 0]), args=(mass, u, index), bounds=bounds)
    return (i, j, res.fun, res.x[-1], res.success, res.x)

def marginalize_mass_u(index):
    param_grid = list(product(range(len(u_bins)), range(len(mass_bins))))
                      
    with Pool() as pool:
        results = list(tqdm(pool.istarmap(partial(marginalize_mass_u_helper, index=index), param_grid), total=len(param_grid)))

    chi2 = np.full((len(u_bins), len(mass_bins)), 1e6)
    margin_u = np.zeros((len(u_bins), len(mass_bins)))
    success = np.zeros((len(u_bins), len(mass_bins)))
    xs = np.zeros((len(u_bins), len(mass_bins), 6))

    for i, j, val, u, suc, x in results:
        chi2[i, j] = val
        margin_u[i, j] = u
        success[i, j] = suc
        xs[i, j] = x

    chi2 = chi2 - np.min(chi2)
    np.save("output/chi2_csi_mass_ue4.npy", chi2)
    np.save("output/chi2_csi_mass_ue4_margin_umu4.npy", margin_u)
    np.save("output/chi2_csi_mass_ue4_success.npy", success)
    np.save("output/chi2_csi_mass_ue4_params.npy", xs)

def marginalize_mass_uu_helper(i, j, k):
    u1 = u_bins[i]
    u2 = u_bins[j]
    mass = mass_bins[k]
    if u1 + u2 > 1:
        return (i, j, k, 1e6, [0, 0, 0, 0, 0], False)
    res = iminuit.minimize(cost, np.asarray([80, 0, 0, 0, 0]), args=(mass, u1, 1, u2))
    return (i, j, k, res.fun, res.x, res.success)

def marginalize_mass_uu():
    param_grid = list(product(range(len(u_bins)), range(len(u_bins)), range(len(mass_bins))))

    with Pool() as pool:
        results = list(tqdm(pool.istarmap(marginalize_mass_uu_helper, param_grid), total=len(param_grid)))
                       
    chi2 = np.full((len(u_bins), len(u_bins), len(mass_bins)), 1e6)
    xs = np.zeros((len(u_bins), len(u_bins), len(mass_bins), 5))
    success = np.zeros((len(u_bins), len(u_bins), len(mass_bins)))

    try:
        np.save("output/results.npy", results)
    except:
        pass

    for i, j, k, val, x, suc in results:
        chi2[i, j, k] = val
        xs[i, j, k] = x
        success[i, j, k] = suc
        
    chi2 = chi2 - np.min(chi2)
    np.save("output/chi2_csi_mass_ue4_umu4.npy", chi2)
    np.save("output/chi2_csi_mass_ue4_umu4_success.npy", success)
    np.save("output/chi2_csi_mass_ue4__umu4_params.npy", xs)

if __name__ == "__main__":
    # main()
    # plot_sin2theta()
    # marginalize_mass_uu()
    # plot_sin2theta_new()
    # marginalize_mass_u(1)
    # plot_u(1)
    plot_sin2theta(1)
    plot_sin2theta(2)
    # plot_sin2theta_mue(1)

    # chi2 = np.load("chi2_10m_nosub.npy")
    # xs = np.load("xs_10m.npy")
    # i = np.searchsorted(u_bins, 0.3)
    # j = np.searchsorted(mass_bins, 3)

    # print(chi2[i, j])
    # print(chi2[i+1, j+1])
    # print(xs[i, j])
