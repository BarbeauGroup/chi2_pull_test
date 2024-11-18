import numpy as np
import timeit
from numba import jit

def num_atoms(params):
    mass = params["detector"]["mass"]
    num_isotopes = len(params["detector"]["isotopes"])
    Da = 1.66e-27

    A = 0
    for isotope in params["detector"]["isotopes"]:
        A += isotope["mass"] * isotope["abundance"]
    
    A /= num_isotopes

    return mass / (A * Da)

def csi_energy_efficiency(x):
    """
    Calculate the CsI detector efficiency.

    Parameters
    ----------
    x : float
        Energy in PE.
    
    Returns
    -------
    float
        The efficiency.
    """
    eps = (1.32045 / (1 + np.exp(-0.285979*(x - 10.8646)))) - 0.333322
    return np.where(x > 60, 0, np.where(eps < 0, 0, eps))

def csi_time_efficiency(t):
    """
    Calculate the efficiency of the CsI detector as a function of time.

    Parameters
    ----------
    t: float
        The time at which the efficiency is calculated in ns.
    
    Returns
    -------
    float
        The efficiency.
    """
    a = 520
    b = 0.0494 / 1000

    return np.where(t < 0, 0, np.where(t < a, 1, np.where(t < 6000, np.exp(-b*(t - a)), 0)))

def create_observables(flux, params, time_offset) -> dict:
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

    # flux is normalized to number of neutrinos per POT so it's not counted here
    pot_per_cm2 = params["beam"]["pot"] / (4 * np.pi * np.power(params["detector"]["distance"], 2))

    # Load in energy binning
    dx = params["detector"]["detector_matrix_dx"]
    observable_bin_arr = np.arange(0, detector_matrix.shape[0] * dx, dx)

    # Do time offset
    new_time_edges = flux["nuE"][0][0][:-1] + time_offset

    # Calculate the efficiency arrays
    energy_efficiency = csi_energy_efficiency(observable_bin_arr)
    time_efficiency = csi_time_efficiency(new_time_edges)

    efficiency = energy_efficiency[:, None] * time_efficiency

    flavor_arr = ["nuE", "nuEBar", "nuMu", "nuMuBar", "nuTau", "nuTauBar", "nuS", "nuSBar"]
    flux_arr = []
    for flavor in flavor_arr:
        flux_arr.append(flux[flavor])

    mass = params["detector"]["mass"]
    num_isotopes = len(params["detector"]["isotopes"])
    Da = 1.66e-27

    A = 0
    for isotope in params["detector"]["isotopes"]:
        A += isotope["mass"] * isotope["abundance"]

    A /= num_isotopes

    num_atoms = mass / (A * Da)

    @jit
    def do_it(flux_arr):
        observables = {}

        for i in range(len(flux_arr)):
            # Load in energy and calculate observable energy before efficiencies
            truth = np.dot(flux_matrix, np.transpose(flux_arr[i][1]))
            observable = np.dot(detector_matrix, truth)

            # Rescale counts
            observable *= num_atoms * pot_per_cm2

            # Apply efficiencies
            post_efficiency = observable * efficiency

            observables[flavor] = ((new_time_edges, observable_bin_arr), post_efficiency)


            return observables
    
    return do_it(flux_arr)