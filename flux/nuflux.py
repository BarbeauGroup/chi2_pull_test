import uproot
import numpy as np

from probabilities import Pab


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

    # Make a tuple of the flux information for each neutrino type
    flux_info = {
        'NuMuEnergy': (anc_energy_centered, anc_keNuMu_values),
        'NuMuTime': (anc_time_centered, anc_tNuMu_values),

        'NuMuBarEnergy': (anc_energy_centered, anc_keNuMuBar_values),
        'NuMuBarTime': (anc_time_centered, anc_tNuMuBar_values),

        'NuEEnergy': (anc_energy_centered, anc_keNuE_values),
        'NuETime': (anc_time_centered, anc_tNuE_values),

        'NuEBarEnergy': (anc_energy_centered, anc_keNuEBar_values),
        'NuEBarTime': (anc_time_centered, anc_tNuEBar_values),

        'NuTauEnergy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
        'NuTauTime': (anc_time_centered, np.zeros(anc_tNuE_values.shape)),

        'NuTauBarEnergy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
        'NuTauBarTime': (anc_time_centered, np.zeros(anc_tNuE_values.shape)),

        'NuSEnergy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
        'NuSTime': (anc_time_centered, np.zeros(anc_tNuE_values.shape)),

        'NuSBarEnergy': (anc_energy_centered, np.zeros(anc_keNuE_values.shape)),
        'NuSBarTime': (anc_time_centered, np.zeros(anc_tNuE_values.shape))
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
    Umu4 = 0.1778/3
    Utau4 = 0.0
    Us4 = 1 - np.abs(Ue4)**2 - np.abs(Umu4)**2 - np.abs(Utau4)**2

    print("Total number of muon neutrinos: ", np.sum(flux['NuMuEnergy'][1]))
    print("Total number of muon anti-neutrinos: ", np.sum(flux['NuMuBarEnergy'][1]))
    print("\n")
    print("Total number of electron neutrinos: ", np.sum(flux['NuEEnergy'][1]))
    print("Total number of electron anti-neutrinos: ", np.sum(flux['NuEBarEnergy'][1]))
    print("\n")
    print("Total number of tau neutrinos: ", np.sum(flux['NuTauEnergy'][1]))
    print("Total number of tau anti-neutrinos: ", np.sum(flux['NuTauBarEnergy'][1]))
    print("\n")
    print("Total number of sterile neutrinos: ", np.sum(flux['NuSEnergy'][1]))
    print("Total number of sterile anti-neutrinos: ", np.sum(flux['NuSBarEnergy'][1]))
    print("\n")
    print("Total number of all neutrinos: ", np.sum(flux['NuMuEnergy'][1]) + np.sum(flux['NuEEnergy'][1]) + np.sum(flux['NuTauEnergy'][1]) + np.sum(flux['NuSEnergy'][1]))

    # Make an empty dictionary to store the oscillated flux information
    oscillated_flux = {}

    P_nue_nue = Pab(flux["NuEEnergy"][0], L, deltam41, 1, 1, Ue4, Umu4, Utau4)
    P_nue_numu = Pab(flux["NuMuEnergy"][0], L, deltam41, 2, 1, Ue4, Umu4, Utau4)
    P_nue_nutau = Pab(flux["NuTauEnergy"][0], L, deltam41, 3, 1, Ue4, Umu4, Utau4)
    P_nue_nus = Pab(flux["NuSEnergy"][0], L, deltam41, 4, 1, Ue4, Umu4, Utau4)

    P_numu_nue = Pab(flux["NuMuEnergy"][0], L, deltam41, 2, 2, Ue4, Umu4, Utau4)
    P_numu_numu = Pab(flux["NuEEnergy"][0], L, deltam41, 1, 2, Ue4, Umu4, Utau4)
    P_numu_nutau = Pab(flux["NuTauEnergy"][0], L, deltam41, 3, 2, Ue4, Umu4, Utau4)
    P_numu_nus = Pab(flux["NuSEnergy"][0], L, deltam41, 4, 2, Ue4, Umu4, Utau4)

    P_nutau_nue = Pab(flux["NuTauEnergy"][0], L, deltam41, 3, 3, Ue4, Umu4, Utau4)
    P_nutau_numu = Pab(flux["NuMuEnergy"][0], L, deltam41, 2, 3, Ue4, Umu4, Utau4)
    P_nutau_nutau = Pab(flux["NuEEnergy"][0], L, deltam41, 1, 3, Ue4, Umu4, Utau4)
    P_nutau_nus = Pab(flux["NuSEnergy"][0], L, deltam41, 4, 3, Ue4, Umu4, Utau4)

    P_nus_nue = Pab(flux["NuSEnergy"][0], L, deltam41, 4, 4, Ue4, Umu4, Utau4)
    P_nus_numu = Pab(flux["NuMuEnergy"][0], L, deltam41, 2, 4, Ue4, Umu4, Utau4)
    P_nus_nutau = Pab(flux["NuTauEnergy"][0], L, deltam41, 3, 4, Ue4, Umu4, Utau4)
    P_nus_nus = Pab(flux["NuEEnergy"][0], L, deltam41, 1, 4, Ue4, Umu4, Utau4)

    print(P_nue_nue )

