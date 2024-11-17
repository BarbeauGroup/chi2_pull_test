import emcee
import numpy as np
import corner
from multiprocessing import Pool
import scipy.optimize as op

from utils.loadparams import load_params
from utils.histograms import rebin_histogram
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt
from stats.fit import chi2_stat, chi2_sys

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from flux.ssb_pdf import make_ssb_pdf
from plotting.observables import analysis_bins, plot_observables
from stats.marginalize import marginalize

import matplotlib.pyplot as plt

import time

import cProfile

# Maybe needed bc of numpy matrix multiplication
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Global variables for data
with open("flux/flux_dict.pkl", "rb") as f:
    flux = np.load(f, allow_pickle=True).item()

params = load_params("config/csi.json")
ssb_dict = make_ssb_pdf(params)
bkd_dict = read_brns_nins_from_txt(params)
data_dict = read_data_from_txt(params)

observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
t_bin_arr = np.asarray(params["analysis"]["time_bins"])


def cost_function_global(x: np.ndarray) -> float:
    """
    # x: (deltam41, Ue4, Umu4, Utau4, a_flux, a_brn, a_nin, a_ssb)
    x: (deltam41, Ue4, Umu4, a_flux, a_brn, a_nin, a_ssb)

    returns: negative chi2 value
    """
    osc_params = x[0:3]

    if(osc_params[0] > 100):
        return -np.inf

    if(osc_params[0] < 0 or osc_params[1] < 0 or osc_params[2] < 0):
        return -np.inf
    
    if(np.sum(osc_params[1:3]) > 1):
        return -np.inf

    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], 0.0] #osc_params[3]]   
    nuisance_params = x[3:]
    nuisance_param_priors = [params["detector"]["systematics"]["flux"],
                            params["detector"]["systematics"]["brn"],
                            params["detector"]["systematics"]["nin"],
                            params["detector"]["systematics"]["ssb"]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)
    osc_obs = create_observables(params=params, flux=oscillated_flux)
    histograms_osc = analysis_bins(observable=osc_obs, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)
    
    return -(chi2_stat(histograms=histograms_osc, nuisance_params=nuisance_params) + chi2_sys(nuisance_params=nuisance_params, nuisance_param_priors=nuisance_param_priors))


def plot():
    params = load_params("config/csi.json")
    data_dict = read_data_from_txt(params)

    # no_pkl = False
    # if no_pkl:
    #     flux = read_flux_from_root(params)
    #     with open("flux/flux_dict.pkl", "wb") as f:
    #         np.save(f, flux)
    # else:
    #     with open("flux/flux_dict.pkl", "rb") as f:
    #         flux = np.load(f, allow_pickle=True).item()

    use_backend = False
    if use_backend:
        x = []
        sampler = emcee.backends.HDFBackend("backend.h5")
        flat_samples = sampler.get_chain(discard=300, thin=50, flat=True)
        for i in range(7):
            mcmc = np.percentile(flat_samples[:, i], 50)
            x.append(mcmc)
    else:
        x = [1.521e+00, 4.174e-01, 4.462e-01, 1.196e-02, -4.793e-03,
                 -4.465e-03, -1.447e-02]
    
    osc_params = x[0:3]
    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], 0.0] #osc_params[3]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)

    un_osc_obs = create_observables(params=params, flux=flux)
    osc_obs = create_observables(params=params, flux=oscillated_flux)

    histograms_unosc = analysis_bins(observable=un_osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)
    histograms_osc = analysis_bins(observable=osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)

    cProfile.run("cost_function_global(x)", "output.prof")    
    # plot_observables(params, histograms_unosc, histograms_osc, x[3:])


if __name__ == "__main__":
    # main()
    plot()
    # cProfile.run("plot()", "output.prof")