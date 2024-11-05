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

    # nu mu
    NuMu = convolved_energy_and_time_of_nu_mu.values()
    anc_keNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()
    anc_keNuMu_values = np.sum(NuMu, axis=0)
    anc_keNuMu_values = anc_keNuMu_values / np.sum(anc_keNuMu_values)
    anc_tNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()
    anc_tNuMu_values = np.sum(NuMu, axis=1)
    anc_tNuMu_values = anc_tNuMu_values / np.sum(anc_tNuMu_values)

    # nu mu bar
    NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()
    anc_keNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()
    anc_keNuMuBar_values = np.sum(NuMuBar, axis=0)
    anc_keNuMuBar_values = anc_keNuMuBar_values / np.sum(anc_keNuMuBar_values)
    anc_tNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()
    anc_tNuMuBar_values = np.sum(NuMuBar, axis=1)
    anc_tNuMuBar_values = anc_tNuMuBar_values / np.sum(anc_tNuMuBar_values)

    # nu e
    NuE = convolved_energy_and_time_of_nu_e.values()
    anc_keNuE_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()
    anc_keNuE_values = np.sum(NuE, axis=0)
    anc_keNuE_values = anc_keNuE_values / np.sum(anc_keNuE_values)
    anc_tNuE_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()
    anc_tNuE_values = np.sum(NuE, axis=1)
    anc_tNuE_values = anc_tNuE_values / np.sum(anc_tNuE_values)

    # nu e bar
    NuEBar = convolved_energy_and_time_of_nu_e_bar.values()
    anc_keNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()
    anc_keNuEBar_values = np.sum(NuEBar, axis=0)
    anc_keNuEBar_values = anc_keNuEBar_values / np.sum(anc_keNuEBar_values)
    anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
    anc_tNuEBar_values = np.sum(NuEBar, axis=1)
    anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)

    anc_energy_centered = (anc_keNuE_edges[1:] + anc_keNuE_edges[:-1]) / 2
    anc_time_centered = (anc_tNuE_edges[1:] + anc_tNuE_edges[:-1]) / 2

    # Make a tuple of the flux information for each neutrino type
    flux_info = {
        'nuMu': {
            'energy': (anc_energy_centered, anc_keNuMu_values),
            'time': (anc_time_centered, anc_tNuMu_values)
        },

        'nuMuBar': {
            'energy': (anc_energy_centered, anc_keNuMuBar_values),
            'time': (anc_time_centered, anc_tNuMuBar_values)
        },

        'nuE': {
            'energy': (anc_energy_centered, anc_keNuE_values),
            'time': (anc_time_centered, anc_tNuE_values)
        },

        'nuEBar': {
            'energy': (anc_energy_centered, anc_keNuEBar_values),
            'time': (anc_time_centered, anc_tNuEBar_values)
        },

        'nuTau': {
            'energy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        },

        'nuTauBar': {
            'energy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        },

        'nuS': {
            'energy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
        },

        'nuSBar': {
            'energy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
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
    Us4 = 1 - np.abs(Ue4)**2 - np.abs(Umu4)**2 - np.abs(Utau4)**2

    # Make an empty dictionary to store the oscillated flux information
    oscillated_flux = {}

    flavors = ["nuE", "nuMu", "nuTau", "nuS"]
    anti_flavors = ["nuEBar", "nuMuBar", "nuTauBar", "nuSBar"]

    for initial_i, (initial_flavor, initial_antiflavor) in enumerate(zip(flavors, anti_flavors)):
        temp = 0
        anti_temp = 0
        prob = 0
        for final_i, (final_flavor, final_antiflavor) in enumerate(zip(flavors, anti_flavors)):
            temp += Pab(flux[final_flavor]["energy"][0], L, deltam41, initial_i+1, final_i+1, Ue4, Umu4, Utau4) * flux[final_flavor]["energy"][1]
            prob += Pab(flux[final_flavor]["energy"][0], L, deltam41, initial_i+1, final_i+1, Ue4, Umu4, Utau4)
            anti_temp += Pab(flux[final_antiflavor]["energy"][0], L, deltam41, initial_i+1,  final_i+1, Ue4, Umu4, Utau4) * flux[final_antiflavor]["energy"][1]
        oscillated_flux[initial_flavor] = temp
        oscillated_flux[initial_antiflavor] = anti_temp

    return oscillated_flux