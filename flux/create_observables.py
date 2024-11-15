import numpy as np
import timeit

def num_atoms(params):
    mass = params["detector"]["mass"]
    num_isotopes = len(params["detector"]["isotopes"])
    Da = 1.66e-27

    A = 0
    for isotope in params["detector"]["isotopes"]:
        A += isotope["mass"] * isotope["abundance"]
    
    A /= num_isotopes

    return mass / (A * Da)

def create_observables(flux, params) -> dict:
    """
    This is the linear algebra step: flux -> true -> reconstructed

    Parameters
    ----------

    Returns
    -------
    dict
        The true observables.
    """

    flux_matrix = np.load(params["detector"]["flux_matrix"])
    detector_matrix = np.load(params["detector"]["detector_matrix"])
    energy_matrix = np.load(params["detector"]["energy_matrix"])
    time_matrix = np.load(params["detector"]["time_matrix"])

    # flux is normalized to number of neutrinos per POT so it's not counted here
    pot_per_cm2 = params["beam"]["pot"] / (4 * np.pi * np.power(params["detector"]["distance"], 2))
    
    observables = {}

    for flavor in flux.keys():
        # Load in energy and calculate observable energy before efficiencies
        truth_level_energy = np.dot(flux_matrix, flux[flavor]["energy"][1])
        observable_energy = np.dot(detector_matrix, truth_level_energy)

        # Load in energy binning
        dx = params["detector"]["detector_matrix_dx"]
        observable_bin_arr = np.arange(0, (len(observable_energy) + 1) * dx, dx)

        # Load in time information
        truth_level_time = flux[flavor]["time"][1]

        # Rescale counts
        observable_energy *= num_atoms(params) * pot_per_cm2
        truth_level_time *= np.sum(truth_level_energy) * num_atoms(params) * pot_per_cm2

        # Calculate some quantities:
        post_efficiency_obs_e = np.dot(energy_matrix, observable_energy)
        post_efficiency_truth_level_time = np.dot(time_matrix, truth_level_time)
        oE_sum = np.sum(observable_energy)
        tt_sum = np.sum(truth_level_time)

        # time averaged efficiency
        if(oE_sum != 0):
            e_frac = np.sum(post_efficiency_obs_e) / oE_sum
        else:
            e_frac = 0

        # energy averaged efficiency
        if(tt_sum != 0):
            t_frac = np.sum(post_efficiency_truth_level_time) / tt_sum
        else:
            t_frac = 0

        observables[flavor] = {
            "energy": (observable_bin_arr, post_efficiency_obs_e * t_frac),
            "time": (flux[flavor]["time"][0], post_efficiency_truth_level_time * e_frac)
        }

    return observables