import emcee
import numpy as np
import corner
from multiprocessing import Pool

from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt
from utils.histograms import rebin_histogram
from stats.fit import chi2_stat, chi2_sys

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins, plot_observables

import cProfile
import pstats


# Maybe needed bc of numpy matrix multiplication
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Global variables for data
with open("flux/flux_dict.pkl", "rb") as f:
    flux = np.load(f, allow_pickle=True).item()

params = load_params("config/csi.json")
bkd_dict = read_brns_nins_from_txt(params)
data_dict = read_data_from_txt(params)

observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
t_bin_arr = np.asarray(params["analysis"]["time_bins"])

brn_e_weights = rebin_histogram(bkd_dict["brn"]["energy"][1], bkd_dict["brn"]["energy"][0], observable_bin_arr) / np.diff(observable_bin_arr)
brn_t_weights = rebin_histogram(bkd_dict["brn"]["time"][1], bkd_dict["brn"]["time"][0], t_bin_arr) / np.diff(t_bin_arr)

nin_e_weights = rebin_histogram(bkd_dict["nin"]["energy"][1], bkd_dict["nin"]["energy"][0], observable_bin_arr) / np.diff(observable_bin_arr)
nin_t_weights = rebin_histogram(bkd_dict["nin"]["time"][1], bkd_dict["nin"]["time"][0], t_bin_arr) / np.diff(t_bin_arr)

new_bkd_dict = {}
new_bkd_dict["brn"] = {
    "energy": np.asarray(brn_e_weights),
    "time": np.asarray(brn_t_weights)
}

new_bkd_dict["nin"] = {
    "energy": np.asarray(nin_e_weights),
    "time": np.asarray(nin_t_weights)
}


def cost_function_global(x: np.ndarray) -> float:
    """
    x: (deltam41, Ue4, Umu4, Utau4, a_flux, a_brn, a_nin, a_ssb)

    returns: chi2 value
    """
    osc_params = x[0:4]

    if(osc_params[0] < 0 or osc_params[1] < 0 or osc_params[2] < 0 or osc_params[3] < 0):
        return np.inf
    
    if(np.sum(np.square(osc_params[1:4])) > 1):
        return np.inf

    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], osc_params[3]]   
    nuisance_params = x[4:]
    nuisance_param_priors = [params["detector"]["systematics"]["flux"],
                            params["detector"]["systematics"]["brn"],
                            params["detector"]["systematics"]["nin"],
                            params["detector"]["systematics"]["ssb"]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)
    osc_obs = create_observables(params=params, flux=oscillated_flux)
    histograms_osc = analysis_bins(observable=osc_obs, bkd_dict=new_bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)
    
    return chi2_stat(histograms=histograms_osc, nuisance_params=nuisance_params) + chi2_sys(nuisance_params=nuisance_params, nuisance_param_priors=nuisance_param_priors)



def main():

    global flux, params, bkd_dict, data_dict  # Declare globals for data

    params = load_params("config/csi.json")
    bkd_dict = read_brns_nins_from_txt(params)
    data_dict = read_data_from_txt(params)

    no_pkl = False
    if no_pkl:
        flux = read_flux_from_root(params)
        with open("flux/flux_dict.pkl", "wb") as f:
            np.save(f, flux)
    else:
        with open("flux/flux_dict.pkl", "rb") as f:
            flux = np.load(f, allow_pickle=True).item()



        

    x = [.1, .1, .1, .1, .1, .1, .1, .1]  # Initial guess




    chi2 = cost_function_global(x)

    print(chi2)





if __name__ == "__main__":
    
    cProfile.run('main()', 'output.prof')