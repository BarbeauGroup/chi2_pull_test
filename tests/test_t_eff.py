
import sys
sys.path.append("./")
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
from stats.likelihood import loglike_stat, loglike_sys


def test_t_eff():
    params = load_params("config/csi.json")
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    flux = read_flux_from_root(params, np.arange(0, 15025, 25))
    # with open("flux/flux_dict.pkl", "wb") as f:
    #     np.save(f, flux)
    
    print(flux)

    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])

    flux_matrix = np.load(params["detector"]["flux_matrix"])
    detector_matrix = np.load(params["detector"]["detector_matrix"])

    # return

    x = [1.521e+00, 4.174e-01, 4.462e-01, 1.196e-02, -4.793e-03,
                 -4.465e-03, -1.447e-02, 58.0]
    
    print(x)
    osc_params = x[0:3]
    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], 0.0] #osc_params[3]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)

    un_osc_obs = create_observables(params=params, flux=flux, time_offset=x[-1], flux_matrix=flux_matrix, detector_matrix=detector_matrix, flavorblind=True)

    print(np.sum(un_osc_obs["combined"][1]))


    # osc_obs = create_observables(params=params, flux=oscillated_flux, time_offset=x[-1], flux_matrix=flux_matrix, detector_matrix=detector_matrix)

    # histograms_unosc = analysis_bins(observable=un_osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=x[-1])
    # histograms_osc = analysis_bins(observable=osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=x[-1])

    # print(cost_function_global(x))

    # plot_observables(params, histograms_unosc, histograms_osc, x[3:-1])


if __name__ == "__main__":
    test_t_eff()