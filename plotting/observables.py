import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from utils.histograms import rebin_histogram

plt.style.use(['science', 'high-contrast'])

def analysis_bins(observable: dict, bkd_dict: dict, data: dict, params: dict, brn_norm: float, nin_norm: float) -> dict:
    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    beam_dict = {}
    for beam_state in ["C", "AC"]:
        pe_hist, _ = np.histogram(data[beam_state]["energy"], bins=observable_bin_arr)
        pe_err = np.sqrt(pe_hist)
        pe_err = np.divide(pe_err, np.diff(observable_bin_arr))
        pe_hist = np.divide(pe_hist, np.diff(observable_bin_arr))

        t_hist, _ = np.histogram(data[beam_state]["time"], bins=t_bin_arr)
        t_err = np.sqrt(t_hist)
        t_err = np.divide(t_err, np.diff(t_bin_arr))
        t_hist = np.divide(t_hist, np.diff(t_bin_arr))

        histograms = {}

        e_weights = pe_hist
        t_weights = t_hist

        beam_dict[beam_state] = {
            "energy": e_weights,
            "time": t_weights
        }
    
    histograms["beam_state"] = beam_dict

    flavor_dict = {}
    for flavor in observable.keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        e_weights = rebin_histogram(observable[flavor]["energy"][1], observable[flavor]["energy"][0], observable_bin_arr) / np.diff(observable_bin_arr)
        t_weights = rebin_histogram(observable[flavor]["time"][1], observable[flavor]["time"][0]/1000, t_bin_arr) / np.diff(t_bin_arr) #TODO: mandate times are same unit

        flavor_dict[flavor] = {
            "energy": e_weights,
            "time": t_weights
        }
    histograms["neutrinos"] = flavor_dict

    final_bkd_dict = {}
    for bkd in bkd_dict.keys():
        if bkd == "brn": norm = brn_norm
        else: norm = nin_norm
        e_weights = norm*rebin_histogram(bkd_dict[bkd]["energy"][1], bkd_dict[bkd]["energy"][0], observable_bin_arr) / np.diff(observable_bin_arr)
        t_weights = norm*rebin_histogram(bkd_dict[bkd]["time"][1], bkd_dict[bkd]["time"][0], t_bin_arr) / np.diff(t_bin_arr)

        final_bkd_dict[bkd] = {
            "energy": e_weights,
            "time": t_weights
        }
    histograms["backgrounds"] = final_bkd_dict
    
    return histograms


def plot_observables(params: dict, histograms_unosc: dict, histograms_osc: dict, alpha: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # stacked histogram
    x = []
    t = []
    e_weights = []
    t_weights = []
    labels = []

    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    # beam state subtraction
    pe_hist = histograms_unosc["beam_state"]["C"]["energy"] - histograms_unosc["beam_state"]["AC"]["energy"] * ( 1 + alpha)
    t_hist = histograms_unosc["beam_state"]["C"]["time"] - histograms_unosc["beam_state"]["AC"]["time"] * ( 1 + alpha)
    pe_err = np.sqrt(histograms_unosc["beam_state"]["C"]["energy"] + histograms_unosc["beam_state"]["AC"]["energy"] * ( 1 + alpha))
    t_err = np.sqrt(histograms_unosc["beam_state"]["C"]["time"] + histograms_unosc["beam_state"]["AC"]["time"] * ( 1 + alpha))

    for bkd in histograms_unosc["backgrounds"].keys():
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(histograms_unosc["backgrounds"][bkd]["energy"])
        t_weights.append(histograms_unosc["backgrounds"][bkd]["time"])

        labels.append(bkd)

    for flavor in histograms_unosc["neutrinos"].keys():
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(histograms_unosc["neutrinos"][flavor]["energy"])
        t_weights.append(histograms_unosc["neutrinos"][flavor]["time"])

        labels.append(flavor)
    
    ax[0].hist(x, bins=observable_bin_arr, weights=e_weights,
                stacked=True, 
                histtype='step',
                edgecolor='black')
    
    ax[0].hist(x, bins=observable_bin_arr, weights=e_weights,
            stacked=True, 
            label=labels,
            alpha=1,
            color=["red", "sienna", "darkorange", "gold", "greenyellow", "seagreen", "mediumturquoise", "dodgerblue"])      

    ax[0].errorbar(x=(observable_bin_arr[1:] + observable_bin_arr[:-1])/2, y=pe_hist, ls="none", yerr=np.sqrt(pe_err), label="C", color="red", marker="x", markersize=5)
    
    ax[0].set_xlabel("Energy [PE]")
    ax[0].set_ylabel("Counts / PE")

    ax[1].hist(t, bins=t_bin_arr, weights=t_weights,
                   stacked=True, 
                    histtype='step',
                    edgecolor='black')

    ax[1].hist(t, bins=t_bin_arr, weights=t_weights,
                stacked=True, 
                label=labels,
                alpha=1,
                color=["red", "sienna", "darkorange", "gold", "greenyellow", "seagreen", "mediumturquoise", "dodgerblue"])
    
    ax[1].errorbar(x=(t_bin_arr[1:] + t_bin_arr[:-1])/2, y=t_hist, ls="none", yerr=np.sqrt(t_err), label="C", color="red", marker="x", markersize=5)
    
    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_ylabel(r"Counts / $\mu$s")


    e_weights = 0
    t_weights = 0

    for bkd in histograms_osc["backgrounds"].keys():
        e_weights += histograms_osc["backgrounds"][bkd]["energy"]
        t_weights += histograms_osc["backgrounds"][bkd]["time"]

    for flavor in histograms_osc["neutrinos"].keys():
        e_weights += histograms_osc["neutrinos"][flavor]["energy"]
        t_weights += histograms_osc["neutrinos"][flavor]["time"]

    ax[0].legend()
    ax[1].legend()
    
    ax[0].errorbar(x=(observable_bin_arr[1:] + observable_bin_arr[:-1])/2,
                y=e_weights,
                xerr=np.diff(observable_bin_arr)/2, 
                label="Oscillated", 
                color="blue", ls="none", 
                marker="x", markersize=5)
    
    ax[1].errorbar(x=(t_bin_arr[1:] + t_bin_arr[:-1])/2,
                y=t_weights,
                xerr=np.diff(t_bin_arr)/2, 
                label="Oscillated", 
                color="blue", ls="none", 
                marker="x", markersize=5)

    ax[0].legend()
    ax[1].legend()

    plt.plot()
    plt.show()

    return