import numpy as np
import timeit
from numba import njit


from utils.histograms import rebin_histogram2d

def create_observables(flux, experiment, nuisance_params, flavorblind=False) -> dict:
    """
    This is the linear algebra step: flux -> true -> reconstructed

    Parameters
    ----------

    Returns
    -------
    dict
        The true observables.
    """
    # flux is normalized to number of neutrinos per POT so it's not counted here
    pot_per_cm2 = experiment.params["beam"]["pot"] / (4 * np.pi * np.power(experiment.params["detector"]["distance"] * 100, 2))

    # Load in energy binning
    dx = experiment.params["detector"]["detector_matrix_dx"]
    observable_bin_arr = np.arange(0, experiment.detector_matrix.shape[0] * dx, dx)

    # Load in time analysis bins
    t_anal_bins = experiment.observable_time_bins

    # Flux energy bins
    flux_energy_bins = np.arange(0, 60, 1) # TODO add to param file? / flux object

    # Do time offset
    if "nu_time_offset" in nuisance_params:
        new_time_edges = flux["nuE"][0][0][:-1] + nuisance_params["nu_time_offset"]
    else:
        new_time_edges = flux["nuE"][0][0][:-1]

    # Calculate the efficiency arrays
    energy_efficiency = experiment.energy_efficiency(observable_bin_arr)
    time_efficiency = experiment.time_efficiency(new_time_edges)
    
    # Add together isotope fluxes somehow
    if flavorblind:
        observables = {
            "combined": [[t_anal_bins*1000., observable_bin_arr], 0]
        }
    else:
        observables = {
            "nuE": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuMu": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuTau": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuEBar": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuMuBar": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuTauBar": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuS": [[t_anal_bins*1000., observable_bin_arr], 0],
            "nuSBar": [[t_anal_bins*1000., observable_bin_arr], 0],
        }

    for isotope in experiment.params["detector"]["isotopes"]:
        recoil_bins = np.linspace(0, isotope["flux_matrix"].shape[0] * experiment.params["detector"]["flux_matrix_dx"], isotope["flux_matrix"].shape[0])

        ff_2 = experiment.form_factor(isotope, recoil_bins, nuisance_params[f"r_n_{experiment.params['name']}"])**2
        # print(isotope["name"])
        # print(ff_2)

        def calculate(flux_object):
            # Apply time efficiency to flux object
            flux_post_te = flux_object * time_efficiency

            # Rebin in time
            flux_rebinned = rebin_histogram2d(flux_post_te, flux_energy_bins, new_time_edges, flux_energy_bins, t_anal_bins*1000.)

            # Load in energy and calculate observable energy before efficiencies
            # observable = experiment.matrix @ flux_object

            recoil_spectrum = isotope["flux_matrix"] @ flux_rebinned # TODO: change flux matrix
            recoil_spectrum_post_ff = recoil_spectrum * ff_2[:, None]
            observable = experiment.detector_matrix @ recoil_spectrum_post_ff

            # Rescale counts 
            observable *= isotope["num_atoms"] * pot_per_cm2 # TODO: Change num atoms

            # Apply energy efficiency
            post_efficiency = observable * energy_efficiency[:, None]

            # print(np.sum(post_efficiency))

            return post_efficiency

        combined_flux = 0
        for flavor in flux.keys():
            if flavorblind:
                if flavor in experiment.params["detector"]["observable_flavors"]:
                    combined_flux += flux[flavor][1]
            else:
                observables[flavor][1] += calculate(flux[flavor][1].T)
                # print(flavor, np.sum(flux[flavor][1]))
        
        if flavorblind:
            observables["combined"][1] += calculate(combined_flux.T)
    
    # e = 0
    # for flavor in observables.keys():
    #     e += np.sum(observables[flavor][1])
    # print("total", e)

    return observables