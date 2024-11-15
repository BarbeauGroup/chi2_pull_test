import uproot
import numpy as np

from utils.histograms import centers_to_edges

from utils.histograms import rebin_histogram

def read_flux_from_root(params: dict) -> dict:
    """
    
    params: dictionary with flux_file key and other info

    returns: dictionary with SNS flux information
    
    """

    filename = params["beam"]["flux_file"]
    paper_rf = uproot.open(filename)

    convolved_energy_and_time_of_nu_mu = paper_rf["convolved_energy_time_of_nu_mu;1"]
    convolved_energy_and_time_of_nu_mu_bar = paper_rf["convolved_energy_time_of_anti_nu_mu;1"]
    convolved_energy_and_time_of_nu_e = paper_rf["convolved_energy_time_of_nu_e;1"]
    convolved_energy_and_time_of_nu_e_bar = paper_rf["convolved_energy_time_of_anti_nu_e;1"]

    # TODO: put in config file
    new_t_edges = np.arange(0, 15010, 10)
    
    # nu e
    NuE = convolved_energy_and_time_of_nu_e.values()
    anc_keNuE_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()[1:60]
    anc_keNuE_values = np.sum(NuE, axis=0)[1:60]
    anc_keNuE_values = (anc_keNuE_values / np.sum(anc_keNuE_values))*params["beam"]["nus_per_pot"]["nuE"]

    anc_tNuE_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()
    anc_tNuE_values = np.sum(NuE, axis=1)
    anc_tNuE_values = rebin_histogram(anc_tNuE_values, anc_tNuE_edges, new_t_edges)
    anc_tNuE_values = anc_tNuE_values / np.sum(anc_tNuE_values)

    # nu e bar
    NuEBar = convolved_energy_and_time_of_nu_e_bar.values()
    anc_keNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()[1:60]
    anc_keNuEBar_values = np.sum(NuEBar, axis=0)[1:60]
    anc_keNuEBar_values = (anc_keNuEBar_values / np.sum(anc_keNuEBar_values))*params["beam"]["nus_per_pot"]["nuEBar"]

    anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
    anc_tNuEBar_values = np.sum(NuEBar, axis=1)
    anc_tNuEBar_values = rebin_histogram(anc_tNuEBar_values, anc_tNuEBar_edges, new_t_edges)
    anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)
    
    # nu mu
    NuMu = convolved_energy_and_time_of_nu_mu.values()
    anc_keNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()[1:60]
    anc_keNuMu_values = np.sum(NuMu, axis=0)[1:60]
    anc_keNuMu_values = (anc_keNuMu_values / np.sum(anc_keNuMu_values))*params["beam"]["nus_per_pot"]["nuMu"]

    anc_tNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()
    anc_tNuMu_values = np.sum(NuMu, axis=1)
    anc_tNuMu_values = rebin_histogram(anc_tNuMu_values, anc_tNuMu_edges, new_t_edges)
    anc_tNuMu_values = anc_tNuMu_values / np.sum(anc_tNuMu_values)

    # nu mu bar
    NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()
    anc_keNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()[1:60]
    anc_keNuMuBar_values = np.sum(NuMuBar, axis=0)[1:60]
    anc_keNuMuBar_values = (anc_keNuMuBar_values / np.sum(anc_keNuMuBar_values))*params["beam"]["nus_per_pot"]["nuMuBar"]

    anc_tNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()
    anc_tNuMuBar_values = np.sum(NuMuBar, axis=1)
    anc_tNuMuBar_values = rebin_histogram(anc_tNuMuBar_values, anc_tNuMuBar_edges, new_t_edges)
    anc_tNuMuBar_values = anc_tNuMuBar_values / np.sum(anc_tNuMuBar_values)

    # Make a tuple of the flux information for each neutrino type
    flux_info = {
        'nuE': {
            'energy': (anc_keNuE_edges, anc_keNuE_values),
            'time': (new_t_edges, anc_tNuE_values)
        },

        'nuEBar': {
            'energy': (anc_keNuEBar_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (new_t_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuMu': {
            'energy': (anc_keNuMu_edges, anc_keNuMu_values),
            'time': (new_t_edges, anc_tNuMu_values)
        },

        'nuMuBar': {
            'energy': (anc_keNuMuBar_edges, anc_keNuMuBar_values),
            'time': (new_t_edges, anc_tNuMuBar_values)
        },

        'nuTau': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (new_t_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuTauBar': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (new_t_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuS': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (new_t_edges, np.zeros(anc_tNuE_values.shape))
        },

        'nuSBar': {
            'energy': (anc_keNuMu_edges, np.zeros(anc_keNuE_values.shape)),
            'time': (new_t_edges, np.zeros(anc_tNuE_values.shape))
        }
    }

    return flux_info

def read_brns_nins_from_txt(params: dict) -> dict:
    """
    
    params: dictionary with hist_file key and other info

    returns: dictionary with SNS flux information
    
    """

    dictionary = {}

    # BRNs 
    brn_pe = np.loadtxt(params["beam"]["brn_energy_file"])
    brn_t = np.loadtxt(params["beam"]["brn_time_file"])

    brn_pe_bins_centers = brn_pe[:60,0]
    brn_pe_bins_edges = centers_to_edges(brn_pe_bins_centers)
    brn_pe_counts = brn_pe[:60,1]
    brn_pe_counts /= np.sum(brn_pe_counts)

    brn_t_bins_centers = brn_t[:,0]
    brn_t_bins_edges = centers_to_edges(brn_t_bins_centers)
    brn_t_counts = brn_t[:,1]
    brn_t_counts /= np.sum(brn_t_counts)

    dictionary["brn"] =  { "energy": (brn_pe_bins_edges, brn_pe_counts),
                            "time": (brn_t_bins_edges, brn_t_counts) }

    nin_pe = np.loadtxt(params["beam"]["nin_energy_file"])
    nin_t = np.loadtxt(params["beam"]["nin_time_file"])

    nin_pe_bins_centers = nin_pe[:60,0]
    nin_pe_bins_edges = centers_to_edges(nin_pe_bins_centers)
    nin_pe_counts = nin_pe[:60,1]
    nin_pe_counts /= np.sum(nin_pe_counts)

    nin_t_bins_centers = nin_t[:,0]
    nin_t_bins_edges = centers_to_edges(nin_t_bins_centers)
    nin_t_counts = nin_t[:,1]
    nin_t_counts /= np.sum(nin_t_counts)

    dictionary["nin"] =  { "energy": (nin_pe_bins_edges, nin_pe_counts),
                            "time": (nin_t_bins_edges, nin_t_counts) }

    return dictionary


def read_data_from_txt(params: dict) -> dict:
    """
    
    params: dictionary with hist_file key and other info

    returns: dictionary with measurement data
    
    """
    dataBeamOnAC = np.loadtxt(params["detector"]["beam_ac_data_file"])
    AC_PE = dataBeamOnAC[:,0]
    AC_t = dataBeamOnAC[:,1]

    dataBeamOnC = np.loadtxt(params["detector"]["beam_c_data_file"])
    C_PE = dataBeamOnC[:,0]
    C_t = dataBeamOnC[:,1]

    # must filter the events that have t > time roi (e.g. 6)
    AC_high_time_idx = np.where(AC_t > params["analysis"]["time_roi"][1])[0]
    AC_PE = np.delete(AC_PE, AC_high_time_idx)
    AC_t = np.delete(AC_t, AC_high_time_idx)

    C_high_time_idx = np.where(C_t > params["analysis"]["time_roi"][1])[0]
    C_PE = np.delete(C_PE, C_high_time_idx)
    C_t = np.delete(C_t, C_high_time_idx)

    # and energy > energy roi (e.g. 60)
    AC_high_energy_idx = np.where(AC_PE > params["analysis"]["energy_roi"][1])[0]
    AC_PE = np.delete(AC_PE, AC_high_energy_idx)
    AC_t = np.delete(AC_t, AC_high_energy_idx)

    C_high_energy_idx = np.where(C_PE > params["analysis"]["energy_roi"][1])[0]
    C_PE = np.delete(C_PE, C_high_energy_idx)
    C_t = np.delete(C_t, C_high_energy_idx)

    return {
        "AC": { "energy": AC_PE, "time": AC_t },
        "C": { "energy": C_PE, "time": C_t }
    }
