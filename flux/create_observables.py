import numpy as np

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

    for flavor in flux.keys():
        num_neutrinos = params["detector"]["nus_per_pot"][flavor] * params["detector"]["pot"] / (4 * np.pi * np.power(params["detector"]["distance"], 2))

        # Load in energy and calculate observable energy before efficiencies
        truth_level_energy = np.dot(flux_matrix, flux[flavor]["energy"][1][1:60])
        observable_energy = np.dot(np.nan_to_num(detector_matrix), truth_level_energy)
        dx = params["detector"]["detector_matrix_dx"]
        observable_bin_arr = np.arange(0, len(observable_energy) * dx, dx)

        # Load in time information
        truth_level_time = flux[flavor]["time"][1]

        # true counts before efficiencies
        N_true = np.sum(observable_energy) * num_atoms(params) * num_neutrinos

        # time averaged efficiency
        if(np.sum(observable_energy) != 0):
            e_frac = np.sum(energy_matrix * observable_energy) / np.sum(observable_energy)
        else:
            e_frac = 0

        # energy averaged efficiency
        if(np.sum(truth_level_time) != 0):
            t_frac = np.sum(time_matrix * truth_level_time) / np.sum(truth_level_time)
        else:
            t_frac = 0

        observables[flavor] = {
            "energy": (observable_bin_arr, np.dot(energy_matrix, observable_energy) * num_neutrinos * num_atoms(params) * t_frac),
            "time": (flux[flavor]["time"][0], np.dot(time_matrix, truth_level_time) * N_true * e_frac)
        }

        print("num neutrinos: ")
        print("Observed counts for ", flavor, ":", np.sum(observables[flavor]["energy"][1]), np.sum(observables[flavor]["time"][1]))
    
    return observables