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

def create_observables(flux, params, time_offset, flux_matrix, detector_matrix, flavorblind=False) -> dict:
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
    
    observables = {}

    combined_flux = 0

    def calculate(flux_object):
        # Load in energy and calculate observable energy before efficiencies
        truth = flux_matrix @ flux_object
        observable = detector_matrix @ truth

        # Rescale counts
        observable *= num_atoms(params) * pot_per_cm2

        # Apply efficiencies
        post_efficiency = observable * efficiency

        return post_efficiency

    for flavor in flux.keys():
        if flavorblind and flavor != "nuS" and flavor != "nuSBar":
            combined_flux += flux[flavor][1]
        
        else:
            observables[flavor] = ((new_time_edges, observable_bin_arr), calculate(flux[flavor][1].mT))
    
    if flavorblind:
        observables = {
            "combined": ((new_time_edges, observable_bin_arr), calculate(combined_flux.mT))
        }

    return observables