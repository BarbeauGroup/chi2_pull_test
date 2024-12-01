from classes.Experiment import Experiment
from classes.Flux import Flux
from transform_functions import csi

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins

from stats.likelihood import loglike_stat, loglike_sys

import numpy as np

def cost(x, flux, experiments, fit_param_keys, mass=None, ue4=None, umu4=None):
    """
    fit_param_keys e.g. flux_time_offset
    """

    fit_params = dict(zip(fit_param_keys, x))
    if mass is None:
        mass = fit_params["mass"]
    if ue4 is None:
        ue4 = fit_params["ue4"]
    if umu4 is None:
        umu4 = fit_params["umu4"]

    ll_stat = 0
    fit_param_priors = {}

    for experiment in experiments:
        for k in experiment.params["detector"]["systematics"].keys():
            v = fit_param_priors.get(k)
            if v is not None and v != experiment.params["detector"]["systematics"][k]:
                raise ValueError("Mismatch in priors for shared nuisance parameters")
            fit_param_priors[k] = experiment.params["detector"]["systematics"][k]

        osc_params = [experiment.params["detector"]["distance"], mass, ue4, umu4, 0.0]

        oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)
        osc_obs = create_observables(flux=oscillated_flux, experiment=experiment, nuisance_params=fit_params, flavorblind=True)
        hist_osc = analysis_bins(observable=osc_obs, experiment=experiment, nuisance_params=fit_params)

        ll_stat += loglike_stat(hist_osc, fit_params)
        
    ll_sys = loglike_sys(fit_params, fit_param_priors)

    return -2 * (ll_stat + ll_sys)


def main():
    flux = Flux("config/ensemble.json")
    CsI = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency)

    print(cost([80, 0, 0, 0, 0], flux.flux, [CsI], ["flux_time_offset", "flux", "brn_csi", "nin_csi", "ssb_csi"], 1, 0, 0))

if __name__ == "__main__":
    main()