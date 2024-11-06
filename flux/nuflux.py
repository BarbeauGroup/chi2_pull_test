import uproot
import numpy as np

from flux.probabilities import Pab, sin2theta


def read_flux_from_root(filename: str) -> dict:
    """
    
    filename: str
        The name of the root file containing the flux information.

    returns: dictionary with SNS flux information
    
    """

    paper_rf = uproot.open(filename)

    convolved_energy_and_time_of_nu_mu = paper_rf["convolved_energy_time_of_nu_mu;1"]
    convolved_energy_and_time_of_nu_mu_bar = paper_rf["convolved_energy_time_of_anti_nu_mu;1"]
    convolved_energy_and_time_of_nu_e = paper_rf["convolved_energy_time_of_nu_e;1"]
    convolved_energy_and_time_of_nu_e_bar = paper_rf["convolved_energy_time_of_anti_nu_e;1"]
    
    # nu e
    NuE = convolved_energy_and_time_of_nu_e.values()
    anc_keNuE_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()[1:60]
    anc_keNuE_values = np.sum(NuE, axis=0)[1:60]
    anc_keNuE_values = anc_keNuE_values / np.sum(anc_keNuE_values)
    anc_tNuE_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()
    anc_tNuE_values = np.sum(NuE, axis=1)
    anc_tNuE_values = anc_tNuE_values / np.sum(anc_tNuE_values)

    # nu e bar
    NuEBar = convolved_energy_and_time_of_nu_e_bar.values()
    anc_keNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()[1:60]
    anc_keNuEBar_values = np.sum(NuEBar, axis=0)[1:60]
    anc_keNuEBar_values = anc_keNuEBar_values / np.sum(anc_keNuEBar_values)
    anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
    anc_tNuEBar_values = np.sum(NuEBar, axis=1)
    anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)
    
    # nu mu
    NuMu = convolved_energy_and_time_of_nu_mu.values()
    anc_keNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()[1:60]
    anc_keNuMu_values = np.sum(NuMu, axis=0)[1:60]
    anc_keNuMu_values = anc_keNuMu_values / np.sum(anc_keNuMu_values)
    anc_tNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()
    anc_tNuMu_values = np.sum(NuMu, axis=1)
    anc_tNuMu_values = anc_tNuMu_values / np.sum(anc_tNuMu_values)

    # nu mu bar
    NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()
    anc_keNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()[1:60]
    anc_keNuMuBar_values = np.sum(NuMuBar, axis=0)[1:60]
    anc_keNuMuBar_values = anc_keNuMuBar_values / np.sum(anc_keNuMuBar_values)
    anc_tNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()
    anc_tNuMuBar_values = np.sum(NuMuBar, axis=1)
    anc_tNuMuBar_values = anc_tNuMuBar_values / np.sum(anc_tNuMuBar_values)

    anc_time_centered = (anc_tNuE_edges[1:] + anc_tNuE_edges[:-1]) / 2

    # Make a tuple of the flux information for each neutrino type
    flux_info = {
        'nuE': {
            'energy': (anc_keNuE_edges, anc_keNuE_values),
            'time': (anc_time_centered, anc_tNuE_values)
        },

        'nuEBar': {
            'energy': (anc_keNuEBar_edges, anc_keNuEBar_values),
            'time': (anc_time_centered, anc_tNuEBar_values)
        },

        'nuMu': {
            'energy': (anc_keNuMu_edges, anc_keNuMu_values),
            'time': (anc_time_centered, anc_tNuMu_values)
        },

        'nuMuBar': {
            'energy': (anc_keNuMuBar_edges, anc_keNuMuBar_values),
            'time': (anc_time_centered, anc_tNuMuBar_values)
        },

        'nuTau': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        },

        'nuTauBar': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        },

        'nuS': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        },

        'nuSBar': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        }
    }

    return flux_info



def oscillate_flux(flux: dict) -> dict:
    """
    flux: dictionary with SNS flux information

    returns: dictionary with oscillated SNS flux information
    """

    ###

    # Oscillation Parameters

    L = 19.3
    deltam41 = 1
    Ue4 = 0.3162
    Umu4 = 0
    Utau4 = 0.0

    # Make an empty dictionary to store the oscillated flux information
    oscillated_flux = {}

    flavors = ["nuE", "nuMu", "nuTau", "nuS"]
    anti_flavors = ["nuEBar", "nuMuBar", "nuTauBar", "nuSBar"]

    for initial_i, (initial_flavor, initial_antiflavor) in enumerate(zip(flavors, anti_flavors)):
        oscillated_energy_flux = 0
        anti_oscillated_energy_flux = 0
        for final_i, (final_flavor, final_antiflavor) in enumerate(zip(flavors, anti_flavors)):
            oscillated_energy_flux += Pab(flux[final_flavor]["energy"][0], L, deltam41, initial_i+1, final_i+1, Ue4, Umu4, Utau4) * flux[final_flavor]["energy"][1]
            anti_oscillated_energy_flux += Pab(flux[final_antiflavor]["energy"][0], L, deltam41, initial_i+1,  final_i+1, Ue4, Umu4, Utau4) * flux[final_antiflavor]["energy"][1]

        if np.sum(flux[initial_flavor]["energy"][1]) == 0:
            time_ratio = 0
        else:
            time_ratio = np.sum(oscillated_energy_flux) / np.sum(flux[initial_flavor]["energy"][1])
        if np.sum(flux[initial_antiflavor]["energy"][1]) == 0:
            anti_time_ratio = 0
        else:
            anti_time_ratio = np.sum(anti_oscillated_energy_flux) / np.sum(flux[initial_antiflavor]["energy"][1])

        oscillated_flux[initial_flavor] = {
            "energy" : (flux[initial_flavor]["energy"][0], oscillated_energy_flux),
            "time" : (flux[initial_flavor]["time"][0], flux[initial_flavor]["time"][1] * time_ratio)
        }
        oscillated_flux[initial_antiflavor] = {
            "energy" : (flux[initial_antiflavor]["energy"][0], anti_oscillated_energy_flux),
            "time" : (flux[initial_antiflavor]["time"][0], flux[initial_antiflavor]["time"][1] * anti_time_ratio)
        }

    return oscillated_flux