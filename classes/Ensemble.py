import sys
import json
import numpy as np
from copy import deepcopy

sys.path.append('../')
from utils.data_loaders import read_flux_from_root

from transform_functions import csi
from transform_functions import pb_glass

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins, project_histograms, plot_histograms

from stats.likelihood import loglike_stat, loglike_sys

class Ensemble:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.params = json.load(f)
        
        time_edges = np.arange(*self.params["time_edges"])

        self.flux = read_flux_from_root(self.params, time_edges)

        self.experiments = []
        self.nuisance_params = []
    
    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def set_nuisance_params(self, nuisance_params):
        self.nuisance_params = nuisance_params

    def oscillated_flux(self, experiment, parameters):
        mass = parameters.get("mass", 0.0)
        ue4 = parameters.get("ue4", 0.0)
        umu4 = parameters.get("umu4", 0.0)

        osc_params = [experiment.params["detector"]["distance"], mass, ue4, umu4, 0.0]

        return oscillate_flux(flux=self.flux, oscillation_params=osc_params)

    def histograms(self, experiment, parameters):
        mass = parameters.get("mass", 0.0)
        ue4 = parameters.get("ue4", 0.0)
        umu4 = parameters.get("umu4", 0.0)

        osc_params = [experiment.params["detector"]["distance"], mass, ue4, umu4, 0.0]

        oscillated_flux = oscillate_flux(flux=self.flux, oscillation_params=osc_params)

        osc_obs = create_observables(flux=oscillated_flux, experiment=experiment, nuisance_params=parameters, flavorblind=False)
        unosc_obs = create_observables(flux=self.flux, experiment=experiment, nuisance_params=parameters, flavorblind=False)

        # print(np.sum(osc_obs["nuE"][1]), np.sum(unosc_obs["nuE"][1]),)

        hist_osc = analysis_bins(observable=osc_obs, experiment=experiment, nuisance_params=parameters)
        hist_unosc = analysis_bins(observable=unosc_obs, experiment=experiment, nuisance_params=parameters)
        
        # print(np.sum(hist_osc["neutrinos"]["nuE"]), np.sum(hist_unosc["neutrinos"]["nuE"]),)

        hist_osc_1d = project_histograms(hist_osc)
        hist_unosc_1d = project_histograms(hist_unosc)

        return hist_osc_1d, hist_unosc_1d
        # plot_csi(experiment.params, hist_unosc_1d, hist_osc_1d, alpha)

    def generate_samples(self, parameters, n_samples):
        fit_param_priors = {}

        for experiment in self.experiments:
            for k in experiment.params["detector"]["systematics"].keys():
                if k not in parameters: continue # skip unused nuisance parameters
                v = fit_param_priors.get(k)
                if v is not None and v != experiment.params["detector"]["systematics"][k]:
                    raise ValueError("Mismatch in priors for shared nuisance parameters")
                fit_param_priors[k] = experiment.params["detector"]["systematics"][k]
        
        samples = []
        for i in range(n_samples):
            sample = []
            for k in parameters:
                prior = fit_param_priors.get(k)
                if hasattr(prior, "__len__"):
                    # print(k, prior)
                    sample.append(np.random.uniform(prior[0], prior[1]))
                else:
                    # print(k, prior)
                    sample.append(np.random.normal(0, prior))
            samples.append(sample)
        
        return samples

    # TODO : pull asimov out of cost function loop
    # If there's no time offset, pull create observables out too
    def cost(self, x, flux, experiments, fit_param_keys, mass=None, ue4=None, umu4=None):
        """
        fit_param_keys e.g. nu_time_offset
        """

        fit_params = dict(zip(fit_param_keys, x))
        if mass is None:
            mass = fit_params["mass"]
        if ue4 is None:
            ue4 = fit_params["ue4"]
        if umu4 is None:
            umu4 = fit_params["umu4"]

        asimov_fit_params = deepcopy(fit_params)
        for k in asimov_fit_params.keys():
            asimov_fit_params[k] = 0

        ll_stat = 0
        fit_param_priors = {}

        for experiment in experiments:
            for k in experiment.params["detector"]["systematics"].keys():
                if k not in fit_param_keys: continue # skip unused nuisance parameters
                v = fit_param_priors.get(k)
                if v is not None and v != experiment.params["detector"]["systematics"][k]:
                    raise ValueError("Mismatch in priors for shared nuisance parameters")
                fit_param_priors[k] = experiment.params["detector"]["systematics"][k]

            osc_params = [experiment.params["detector"]["distance"], mass, ue4, umu4, 0.0]

            oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)

            osc_obs = create_observables(flux=oscillated_flux, experiment=experiment, nuisance_params=fit_params, flavorblind=True)
            asimov = create_observables(flux=flux, experiment=experiment, nuisance_params=asimov_fit_params, flavorblind=True)
            
            hists = analysis_bins(observable=osc_obs, experiment=experiment, nuisance_params=fit_params, asimov=asimov)

            ll_stat += loglike_stat(hists, fit_params, experiment.params["name"])
            
        ll_sys = loglike_sys(fit_params, fit_param_priors)

        return -2 * (ll_stat + ll_sys)
    

    def __call__(self, x, mass=None, u1=None, u2=None, sinmue=None):
        if mass is None:
            mass = x[0]
            x = x[1:]
        if u1 is None:
            u1 = x[0]
            x = x[1:]
        if u2 is None:
            if sinmue is None:
                u2 = x[0]
                x = x[1:]
            else:
                u2 = sinmue / (4 * u1)
                
        # print(mass, u1, u2, sinmue)
        return self.cost(x, self.flux, self.experiments, self.nuisance_params, mass, u1, u2)

