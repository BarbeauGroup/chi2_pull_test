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
    
    observables = {}

    tmp = 0
    for flavor in flux.keys():

        # flux is normalized to number of neutrinos per POT so it's not counted here
        num_neutrinos = params["beam"]["pot"] / (4 * np.pi * np.power(params["detector"]["distance"], 2))

        # Load in energy and calculate observable energy before efficiencies
        truth_level_energy = np.dot(flux_matrix, flux[flavor]["energy"][1])
        observable_energy = np.dot(np.nan_to_num(detector_matrix), truth_level_energy)
        dx = params["detector"]["detector_matrix_dx"]
        observable_bin_arr = np.arange(0, (len(observable_energy) + 1) * dx, dx)

        # Load in time information
        truth_level_time = flux[flavor]["time"][1]

        # Calculate some quantities:
        Em_oE = np.dot(energy_matrix, observable_energy)
        tm_tt = np.dot(time_matrix, truth_level_time)
        oE_sum = np.sum(observable_energy)
        tt_sum = np.sum(truth_level_time)

        # Timing
        # print("Em_oE", timeit.timeit(lambda: np.dot(energy_matrix, observable_energy), number=1000)*1000, "us")
        # print("tm_tt", timeit.timeit(lambda: np.dot(time_matrix, truth_level_time), number=1000)*1000, "us")
        # print("oE_sum", timeit.timeit(lambda: np.sum(observable_energy), number=1000)*1000, "us")
        # print("tt_sum", timeit.timeit(lambda: np.sum(truth_level_time), number=1000)*1000, "us")

        # true counts before efficiencies
        N_true = oE_sum * num_atoms(params) * num_neutrinos

        # time averaged efficiency
        if(oE_sum != 0):
            e_frac = np.sum(Em_oE) / oE_sum
        else:
            e_frac = 0

        # energy averaged efficiency
        if(tt_sum != 0):
            t_frac = np.sum(tm_tt) / tt_sum
        else:
            t_frac = 0

        observables[flavor] = {
            "energy": (observable_bin_arr, Em_oE * num_neutrinos * num_atoms(params) * t_frac),
            "time": (flux[flavor]["time"][0], tm_tt * N_true * e_frac)
        }

        # print("Observed counts for ", flavor, ":", np.sum(observables[flavor]["energy"][1]), np.sum(observables[flavor]["time"][1]))
    
    return observables