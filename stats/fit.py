import numpy as np

from stats.chi2 import chi2_stat, chi2_sys

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins

def cost_function(x: np.ndarray, flux: dict, params: dict, bkd_dict: dict, data_dict: dict) -> float:
    """
    x: (deltam41, Ue4, Umu4, Utau4, a_flux, a_brn, a_nin, a_ssb)
    flux: flux dictionary
    params: dictionary of parameters
    bkd_dict: dictionary of background histograms
    data_dict: dictionary of data histograms

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
    histograms_osc = analysis_bins(observable=osc_obs, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)
    
    return chi2_stat(histograms=histograms_osc, nuisance_params=nuisance_params) + chi2_sys(nuisance_params=nuisance_params, nuisance_param_priors=nuisance_param_priors)