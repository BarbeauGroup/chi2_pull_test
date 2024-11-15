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
    NuE = convolved_energy_and_time_of_nu_e.values()[:, 1:60]
    NuE /= np.sum(NuE)
    NuE *= params["beam"]["nus_per_pot"]["nuE"]
    NuE_e_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()[1:60]
    NuE_t_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()

    # nu e bar
    NuEBar = convolved_energy_and_time_of_nu_e_bar.values()[:, 1:60]
    NuEBar /= np.sum(NuEBar)
    NuEBar *= params["beam"]["nus_per_pot"]["nuEBar"]
    NuEBar_e_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()[1:60]
    NuEBar_t_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()

    # nu mu
    NuMu = convolved_energy_and_time_of_nu_mu.values()[:, 1:60]
    NuMu /= np.sum(NuMu)
    NuMu *= params["beam"]["nus_per_pot"]["nuMu"]
    NuMu_e_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()[1:60]
    NuMu_t_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()

    # nu mu bar
    NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()[:, 1:60]
    NuMuBar /= np.sum(NuMuBar)
    NuMuBar *= params["beam"]["nus_per_pot"]["nuMuBar"]
    NuMuBar_e_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()[1:60]
    NuMuBar_t_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()

    # Make a tuple of the flux information for each neutrino type
    return {
        'nuE': ((NuE_t_edges, NuE_e_edges), NuE),
        'nuEBar': ((NuMu_t_edges, NuMu_e_edges), np.zeros_like(NuMu)),
        'nuMu': ((NuMu_t_edges, NuMu_e_edges), NuMu),
        'nuMuBar': ((NuMuBar_t_edges, NuMuBar_e_edges), NuMuBar),
        'nuTau': ((NuMu_t_edges, NuMu_e_edges), np.zeros_like(NuMu)),
        'nuTauBar': ((NuMu_t_edges, NuMu_e_edges), np.zeros_like(NuMu)),
        'nuS': ((NuMu_t_edges, NuMu_e_edges), np.zeros_like(NuMu)),
        'nuSBar': ((NuMu_t_edges, NuMu_e_edges), np.zeros_like(NuMu))
    }

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
