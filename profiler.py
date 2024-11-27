import iminuit
from scipy.optimize import Bounds

from tqdm import tqdm
import numpy as np
import os
from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt

from flux.probabilities import sin2theta

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from flux.ssb_pdf import make_ssb_pdf
from plotting.observables import analysis_bins, plot_observables, project_histograms, plot_observables2d
from plotting.posteriors import plot_posterior, plot_2dposterior
from stats.likelihood import loglike_stat, loglike_sys

import matplotlib.pyplot as plt

# Global variables for data
params = load_params("config/csi.json")

pkl = True
new_time_edges = np.arange(0, 6625, 125)
if not pkl or not os.path.exists("flux/flux_dict.pkl"):
    flux = read_flux_from_root(params, new_time_edges)
    with open("flux/flux_dict.pkl", "wb") as f:
        np.save(f, flux)
else:
    with open("flux/flux_dict.pkl", "rb") as f:
        flux = np.load(f, allow_pickle=True).item()
    if(flux["nuE"][0][0] != new_time_edges).all():
        flux = read_flux_from_root(params, new_time_edges)
        with open("flux/flux_dict.pkl", "wb") as f:
            np.save(f, flux)

ssb_dict = make_ssb_pdf(params)
bkd_dict = read_brns_nins_from_txt(params)
data_dict = read_data_from_txt(params)

observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
t_bin_arr = np.asarray(params["analysis"]["time_bins"])
flux_matrix = np.load(params["detector"]["flux_matrix"])
detector_matrix = np.load(params["detector"]["detector_matrix"])
matrix = detector_matrix @ flux_matrix


def cost(x, mass, u, index):
    """
    x is [other_u, time_offset, a_flux, a_brn, a_nin, a_ssb]
    index is 1 for Ue4, 2 for Umu4, 3 for Utau4
    """

    time_offset = x[1]

    osc_params = [params["detector"]["distance"]/100., mass, 0, 0, 0.0]
    if index == 1:
        osc_params[2] = u
        osc_params[3] = x[0]
    elif index == 2:
        osc_params[2] = x[0]
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

    return -2 * (loglike_stat(hist_osc, x[2:]) + loglike_sys(x[2:], nuisance_param_priors))
    

def main(): 
    u_bins = np.linspace(0, 1, 75)
    mass_bins = np.linspace(0, 50, 80)
    chi2 = np.zeros((len(u_bins), len(mass_bins)))
    margin_u = np.zeros((len(u_bins), len(mass_bins)))

    for i, u in tqdm(enumerate(u_bins)):
        for j, mass in tqdm(enumerate(mass_bins), leave=False):
            bounds = Bounds([0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [1 - u, np.inf, np.inf, np.inf, np.inf, np.inf], keep_feasible=True)
            res = iminuit.minimize(cost, np.asarray([0, 80, 0, 0, 0, 0]), args=(mass, u, 2), bounds=bounds)
            chi2[i, j] = res.fun
            margin_u[i, j] = res.x[0]
            # print(f"mass: {mass}, u: {u}, chi2: {res.fun}, x: {res.x}, success: {res.success}")
    

    min_chi2 = np.min(chi2)
    chi2 = chi2 - min_chi2
    np.save("chi2.npy", chi2)
    np.save("margin_u.npy", margin_u)
    return
    chi2 = np.load("chi2.npy")
    margin_u = np.load("margin_u.npy")
    min_point = np.unravel_index(np.argmin(chi2, axis=None), chi2.shape)

    # plot chi2
    # u_bins = np.linspace(0, 0.5, 50)
    # sin_bins = sin2theta(2, 2, 0, u_bins, 0)
    # u_bins = np.log10(u_bins)
    # mass_bins = np.log10(mass_bins)
    xv, yv = np.meshgrid(u_bins, mass_bins)

    fig, ax = plt.subplots()
    # ctr = ax.contourf(xv, yv, chi2.T, algorithm='serial') # levels=[0, 2.3, 1000], colors=['white', 'grey'],
    # cbar = fig.colorbar(ctr)
    # cbar.ax.set_ylabel('Chi2')

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # # ax.set_xlabel(r"$\sin^2\theta_{\mu\mu}$")
    # ax.set_xlabel(r"$|U_{e 4}|^2$")
    # ax.set_ylabel(r"$\Delta m^2_{41}$")
    # ax.set_xlim(0.01, 1)
    # ax.set_ylim(10**-0.5, 10**1.5)
    # # ax.plot(0.12, 3.3, 'ro')
    # # ax.plot(u_bins[min_point[0]], mass_bins[min_point[1]], 'kx')
    sin2 = sin2theta(1, 2, u_bins, margin_u, 0)
    ctr = ax.matshow(sin2.T)
    cbar = fig.colorbar(ctr)
    cbar.ax.set_ylabel(r"$\sin^2\theta_{e\mu}$")
    plt.plot()
    plt.show()

if __name__ == "__main__":
    main()
