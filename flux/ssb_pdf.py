import numpy as np
from utils.histograms import rebin_histogram, centers_to_edges

import matplotlib.pyplot as plt

def make_ssb_pdf(params: dict) -> dict:
    """
    params: dictionary with SSB information

    returns: tuple with SSB PDF information
    """

    ssb_pdf = {}


    # Energy PDF is just the AC data
    energy_bins = np.asarray(params["analysis"]["energy_bins"])

    dataBeamOnAC = np.loadtxt(params["detector"]["beam_ac_data_file"])
    AC_PE = dataBeamOnAC[:,0]
    AC_t = dataBeamOnAC[:,1]

    # must filter the events that have t > time roi (e.g. 6)
    AC_high_time_idx = np.where(AC_t > params["analysis"]["time_roi"][1])[0]
    AC_PE = np.delete(AC_PE, AC_high_time_idx)

    # and energy > energy roi (e.g. 60)
    AC_high_energy_idx = np.where(AC_PE > params["analysis"]["energy_roi"][1])[0]
    AC_PE = np.delete(AC_PE, AC_high_energy_idx)

    AC_PE_hist, _ = np.histogram(AC_PE, bins=energy_bins)

    ssb_pdf["energy"] = (energy_bins, AC_PE_hist)


    # Time PDF is an analytic function
    time_bins = np.asarray(params["analysis"]["time_bins"])
    k = 0.0494
    def exp_decay(t, k):
        return k * np.exp(-k*t)
    
    t = np.linspace(0, 6, 100)
    y = exp_decay(t, k)
    y /= np.sum(y)
    t_edges = centers_to_edges(t)

    ssb_pdf["time"] = (time_bins, rebin_histogram(y, t_edges, time_bins))

    return ssb_pdf

    # # plot the PDFs
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].step(energy_bins[:-1], AC_PE_hist/np.diff(energy_bins), where="mid")
    # ax[0].set_xlabel("Energy [PE]")
    # ax[0].set_ylabel("Counts")
    # ax[0].set_title("Energy PDF")

    # ax[1].step(time_bins[:-1], ssb_pdf["time"][1]/np.diff(time_bins), where="mid")
    # ax[1].set_xlabel("Time [us]")
    # ax[1].set_ylabel("Counts")
    # ax[1].set_title("Time PDF")

    # plt.show()



