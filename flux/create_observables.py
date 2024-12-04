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
    observable_bin_arr = np.arange(0, experiment.matrix.shape[0] * dx, dx)

    # Load in time analysis bins
    t_anal_bins = experiment.observable_time_bins

    # Flux energy bins
    flux_energy_bins = np.arange(0, 60, 1) # TODO add to param file? / flux object

    # Do time offset
    if "flux_time_offset" in nuisance_params:
        new_time_edges = flux["nuE"][0][0][:-1] + nuisance_params["flux_time_offset"]
    else:
        new_time_edges = flux["nuE"][0][0][:-1]

    # Calculate the efficiency arrays
    energy_efficiency = experiment.energy_efficiency(observable_bin_arr)
    time_efficiency = experiment.time_efficiency(new_time_edges)
    
    observables = {}

    combined_flux = 0

    def calculate(flux_object):
        # Apply time efficiency to flux object
        flux_object *= time_efficiency

        # Rebin in time
        flux_object = rebin_histogram2d(flux_object, flux_energy_bins, new_time_edges, flux_energy_bins, t_anal_bins*1000.)

        # Load in energy and calculate observable energy before efficiencies
        observable = experiment.matrix @ flux_object

        # Rescale counts
        observable *= experiment.num_atoms * pot_per_cm2

        # Apply energy efficiency
        post_efficiency = observable * energy_efficiency[:, None]

        return post_efficiency

    for flavor in flux.keys():
        if flavorblind:
            if flavor in experiment.params["detector"]["observable_flavors"]:
                combined_flux += flux[flavor][1]
        else:
            observables[flavor] = ((t_anal_bins*1000., observable_bin_arr), calculate(flux[flavor][1].T))
    
    if flavorblind:
        observables = {
            "combined": ((t_anal_bins*1000., observable_bin_arr), calculate(combined_flux.T))
        }

    return observables