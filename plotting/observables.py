import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from utils.histograms import rebin_histogram

plt.style.use(['science'])

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

        # e_weights = norm*rebin_histogram(bkd_dict[bkd]["energy"][1], bkd_dict[bkd]["energy"][0], observable_bin_arr) / np.diff(observable_bin_arr)
        # t_weights = norm*rebin_histogram(bkd_dict[bkd]["time"][1], bkd_dict[bkd]["time"][0], t_bin_arr) / np.diff(t_bin_arr)

        e_weights = norm*bkd_dict[bkd]["energy"]
        t_weights = norm*bkd_dict[bkd]["time"]

        final_bkd_dict[bkd] = {
            "energy": e_weights,
            "time": t_weights
        }
    histograms["backgrounds"] = final_bkd_dict

    
    return histograms


def plot_observables(params: dict, histograms_unosc: dict, histograms_osc: dict, alpha) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]

    # stacked histogram
    x = []
    t = []
    e_weights = []
    t_weights = []
    labels = []

    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    # beam state subtraction
    pe_hist = histograms_unosc["beam_state"]["C"]["energy"] - histograms_unosc["beam_state"]["AC"]["energy"] * ( 1 + alpha[3])
    t_hist = histograms_unosc["beam_state"]["C"]["time"] - histograms_unosc["beam_state"]["AC"]["time"] * ( 1 + alpha[3])
    pe_err = np.sqrt(histograms_unosc["beam_state"]["C"]["energy"] + histograms_unosc["beam_state"]["AC"]["energy"] * ( 1 + alpha[3]))
    t_err = np.sqrt(histograms_unosc["beam_state"]["C"]["time"] + histograms_unosc["beam_state"]["AC"]["time"] * ( 1 + alpha[3]))

    # 3+1 model
    for bkd in histograms_osc["backgrounds"].keys():
        if(bkd == "brn"): scale = alpha[1]
        if(bkd == "nin"): scale = alpha[2]
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(histograms_osc["backgrounds"][bkd]["energy"] * (1 + scale))
        t_weights.append(histograms_osc["backgrounds"][bkd]["time"] * (1 + scale))

        labels.append(bkd)

    for flavor in histograms_osc["neutrinos"].keys():
        # don't plot empty flavors
        if np.sum(histograms_osc["neutrinos"][flavor]["energy"]) == 0: continue
        if np.sum(histograms_osc["neutrinos"][flavor]["time"]) == 0: continue
        scale = alpha[0]
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(histograms_osc["neutrinos"][flavor]["energy"] * (1 + scale))
        t_weights.append(histograms_osc["neutrinos"][flavor]["time"] * (1 + scale))

        labels.append(flavor)
    
    ax[0].hist(x, bins=observable_bin_arr, weights=e_weights,
                stacked=True, 
                histtype='step',
                edgecolor='black')
    
    ax[0].hist(x, bins=observable_bin_arr, weights=e_weights,
            stacked=True, 
            label=labels,
            alpha=1,
            color=colors[:len(labels)])      

    ax[1].hist(t, bins=t_bin_arr, weights=t_weights,
                   stacked=True, 
                    histtype='step',
                    edgecolor='black')

    ax[1].hist(t, bins=t_bin_arr, weights=t_weights,
                stacked=True, 
                label=labels,
                alpha=1,
                color=colors[:len(labels)])
    
    # Standard Model

    e_weights = 0
    t_weights = 0

    for bkd in histograms_unosc["backgrounds"].keys():
        if bkd == "brn": scale = alpha[1]
        if bkd == "nin": scale = alpha[2]
        e_weights += histograms_unosc["backgrounds"][bkd]["energy"] * (1 + scale)
        t_weights += histograms_unosc["backgrounds"][bkd]["time"] * (1 + scale)

    for flavor in histograms_unosc["neutrinos"].keys():
        scale = alpha[0]
        e_weights += histograms_unosc["neutrinos"][flavor]["energy"] * (1 + scale)
        t_weights += histograms_unosc["neutrinos"][flavor]["time"] * (1 + scale)

    ax[0].hist(observable_bin_arr[:-1], bins=observable_bin_arr, weights=e_weights, histtype='step', linestyle='dashed', label="Standard Model", color="grey")
    ax[1].hist(t_bin_arr[:-1], bins=t_bin_arr, weights=t_weights, histtype='step', linestyle='dashed', label="Standard Model", color="grey")

    # data
    ax[0].errorbar(x=(observable_bin_arr[1:] + observable_bin_arr[:-1])/2, y=pe_hist, ls="none", yerr=np.sqrt(pe_err), label="C", color="black", marker="x", markersize=5)
    ax[1].errorbar(x=(t_bin_arr[1:] + t_bin_arr[:-1])/2, y=t_hist, ls="none", yerr=np.sqrt(t_err), label="C", color="black", marker="x", markersize=5)
    
    ax[0].set_xlabel("Energy [PE]")
    ax[0].set_ylabel("Counts / PE")

    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_ylabel(r"Counts / $\mu$s")

    ax[1].set_xscale('function', functions=(lambda x: np.where(x < 1, x, (x - 1) / 3 + 1),
                                            lambda x: np.where(x < 1, x, 3 * (x - 1) + 1)))

    ax[0].legend()
    ax[1].legend()

    plt.plot()
    plt.show()

    return