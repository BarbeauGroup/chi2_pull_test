import uproot
import numpy as np


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
    anc_keNuEBar_values = anc_keNuEBar_values / np.sum(anc_keNuEBar_values) * 0.0001 / 0.0848
    anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
    anc_tNuEBar_values = np.sum(NuEBar, axis=1)
    anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)  * 0.0001 / 0.0848


    anc_energy_centered = (anc_keNuE_edges[1:] + anc_keNuE_edges[:-1]) / 2
    anc_time_centered = (anc_tNuE_edges[1:] + anc_tNuE_edges[:-1]) / 2


    print(anc_keNuMu_edges.shape, anc_energy_centered.shape)

    # Make a tuple of the flux information for each neutrino type
    flux_info = {
        'NuMuEnergy': (anc_energy_centered, anc_keNuMu_values),
        'NuMuTime': (anc_time_centered, anc_tNuMu_values),

        'NuMuBarEnergy': (anc_keNuMuBar_edges, anc_keNuMuBar_values),
        'NuMuBarTime': (anc_tNuMuBar_edges, anc_tNuMuBar_values),

        'NuEEnergy': (anc_keNuE_edges, anc_keNuE_values),
        'NuETime': (anc_tNuE_edges, anc_tNuE_values),

        'NuEBarEnergy': (anc_keNuEBar_edges, anc_keNuEBar_values),
        'NuEBarTime': (anc_tNuEBar_edges, anc_tNuEBar_values),

        'NuTauEnergy': (anc_keNuE_edges, np.zeros(anc_keNuE_edges.shape)),
        'NuTauTime': (anc_tNuE_edges, np.zeros(anc_tNuE_edges.shape)),

        'NuTauBarEnergy': (anc_keNuEBar_edges, np.zeros(anc_keNuEBar_edges.shape)),
        'NuTauBarTime': (anc_tNuEBar_edges, np.zeros(anc_tNuEBar_edges.shape))
    }

    return flux_info



