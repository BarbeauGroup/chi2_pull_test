import uproot
import numpy as np

def read_flux_from_root(params: dict) -> dict:
    """
    
    params: dictionary with flux_file key and other info

    returns: dictionary with SNS flux information
    
    """

    filename = params["flux_file"]
    paper_rf = uproot.open(filename)

    convolved_energy_and_time_of_nu_mu = paper_rf["convolved_energy_time_of_nu_mu;1"]
    convolved_energy_and_time_of_nu_mu_bar = paper_rf["convolved_energy_time_of_anti_nu_mu;1"]
    convolved_energy_and_time_of_nu_e = paper_rf["convolved_energy_time_of_nu_e;1"]
    convolved_energy_and_time_of_nu_e_bar = paper_rf["convolved_energy_time_of_anti_nu_e;1"]
    
    # nu e
    NuE = convolved_energy_and_time_of_nu_e.values()
    anc_keNuE_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()[1:60]
    anc_keNuE_values = np.sum(NuE, axis=0)[1:60]
    anc_keNuE_values = (anc_keNuE_values / np.sum(anc_keNuE_values))*params["detector"]["nus_per_pot"]["nuE"]
    anc_tNuE_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()
    anc_tNuE_values = np.sum(NuE, axis=1)
    anc_tNuE_values = anc_tNuE_values / np.sum(anc_tNuE_values)

    # nu e bar
    NuEBar = convolved_energy_and_time_of_nu_e_bar.values()
    anc_keNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()[1:60]
    anc_keNuEBar_values = np.sum(NuEBar, axis=0)[1:60]
    anc_keNuEBar_values = (anc_keNuEBar_values / np.sum(anc_keNuEBar_values))*params["detector"]["nus_per_pot"]["nuEBar"]
    anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
    anc_tNuEBar_values = np.sum(NuEBar, axis=1)
    anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)
    
    # nu mu
    NuMu = convolved_energy_and_time_of_nu_mu.values()
    anc_keNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()[1:60]
    anc_keNuMu_values = np.sum(NuMu, axis=0)[1:60]
    anc_keNuMu_values = (anc_keNuMu_values / np.sum(anc_keNuMu_values))*params["detector"]["nus_per_pot"]["nuMu"]
    anc_tNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()
    anc_tNuMu_values = np.sum(NuMu, axis=1)
    anc_tNuMu_values = anc_tNuMu_values / np.sum(anc_tNuMu_values)

    # nu mu bar
    NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()
    anc_keNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()[1:60]
    anc_keNuMuBar_values = np.sum(NuMuBar, axis=0)[1:60]
    anc_keNuMuBar_values = (anc_keNuMuBar_values / np.sum(anc_keNuMuBar_values))*params["detector"]["nus_per_pot"]["nuMuBar"]
    anc_tNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()
    anc_tNuMuBar_values = np.sum(NuMuBar, axis=1)
    anc_tNuMuBar_values = anc_tNuMuBar_values / np.sum(anc_tNuMuBar_values)

    # Make a tuple of the flux information for each neutrino type
    flux_info = {
        'nuE': {
            'energy': (anc_keNuE_edges, anc_keNuE_values),
            'time': (anc_tNuE_edges, anc_tNuE_values)
        },

        'nuEBar': {
            'energy': (anc_keNuEBar_edges, anc_keNuEBar_values),
            'time': (anc_tNuEBar_edges, anc_tNuEBar_values)
        },

        'nuMu': {
            'energy': (anc_keNuMu_edges, anc_keNuMu_values),
            'time': (anc_tNuMu_edges, anc_tNuMu_values)
        },

        'nuMuBar': {
            'energy': (anc_keNuMuBar_edges, anc_keNuMuBar_values),
            'time': (anc_tNuMuBar_edges, anc_tNuMuBar_values)
        },

        'nuTau': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_tNuE_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuTauBar': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_tNuE_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuS': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_tNuE_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuSBar': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (anc_tNuE_edges, np.zeros(anc_tNuE_values.shape))
        }
    }

    return flux_info