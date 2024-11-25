import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scienceplots
import numpy as np
from scipy.special import gammaln

from utils.histograms import rebin_histogram, rebin_histogram2d

plt.style.use(['science'])

def analysis_bins(observable: dict, ssb_dict: dict, bkd_dict: dict, data: dict, params: dict, ssb_norm: float, brn_norm: float, nin_norm: float, time_offset: float) -> dict:
    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    histograms = {}
    
    # normalized_ssb_dict = {}
    # for hist in ["energy", "time"]:
    #     normalized_ssb_dict[hist] = ssb_dict[hist] * ssb_norm
    # Constrct 2d histogram of ssb
    ssb_time_pdf = ssb_dict["time"] / np.sum(ssb_dict["time"])
    normalized_ssb = ssb_norm * np.outer(ssb_dict["energy"], ssb_time_pdf) # energy is already normalized to 1 

    histograms["ssb"] = normalized_ssb

    
    beam_dict = {}
    for beam_state in ["C", "AC"]:
        # pe_hist, _ = np.histogram(data[beam_state]["energy"], bins=observable_bin_arr)
        # t_hist, _ = np.histogram(data[beam_state]["time"], bins=t_bin_arr)

        data_hist, _ = np.histogramdd([data[beam_state]["energy"], data[beam_state]["time"]], bins=[observable_bin_arr, t_bin_arr])

        beam_dict[beam_state] = data_hist
    histograms["beam_state"] = beam_dict

    flavor_dict = {}
    for flavor in observable.keys():
        # e_weights = rebin_histogram(np.sum(observable[flavor][1], axis=1), observable[flavor][0][1], observable_bin_arr) 
        # t_weights = rebin_histogram(np.sum(observable[flavor][1], axis=0), observable[flavor][0][0]/1000, t_bin_arr)

        weights = rebin_histogram2d(observable[flavor][1], observable[flavor][0][1], observable[flavor][0][0]/1000, observable_bin_arr, t_bin_arr)

        flavor_dict[flavor] = weights
    histograms["neutrinos"] = flavor_dict

    final_bkd_dict = {}
    for bkd in bkd_dict.keys():
        if bkd == "brn": norm = brn_norm
        else: norm = nin_norm

        e_weights = bkd_dict[bkd]["energy"]
        t_weights = rebin_histogram(bkd_dict[bkd]["time"][1], bkd_dict[bkd]["time"][0] + time_offset/1000., t_bin_arr)

        energy_pdf = e_weights / np.sum(e_weights)
        time_pdf = t_weights / np.sum(t_weights)
        weights = norm * np.outer(energy_pdf, time_pdf)

        final_bkd_dict[bkd] = weights
    histograms["backgrounds"] = final_bkd_dict

    return histograms

def project_histograms(histograms: dict) -> dict:
    # project in energy and time and return dictionary with energy and time keys
    projected_histograms = {}

    ssb_e = np.sum(histograms["ssb"], axis=1)
    ssb_t = np.sum(histograms["ssb"], axis=0)
    projected_histograms["ssb"] = {"energy": ssb_e, "time": ssb_t}

    beam_state_hists = {}
    for beam_state in histograms["beam_state"].keys():
        e = np.sum(histograms["beam_state"][beam_state], axis=1)
        t = np.sum(histograms["beam_state"][beam_state], axis=0)
        beam_state_hists[beam_state] = {"energy": e, "time": t}
    projected_histograms["beam_state"] = beam_state_hists

    neutrino_hists = {}
    for flavor in histograms["neutrinos"].keys():
        e = np.sum(histograms["neutrinos"][flavor], axis=1)
        t = np.sum(histograms["neutrinos"][flavor], axis=0)
        neutrino_hists[flavor] = {"energy": e, "time": t}
    projected_histograms["neutrinos"] = neutrino_hists

    background_hists = {}
    for bkd in histograms["backgrounds"].keys():
        e = np.sum(histograms["backgrounds"][bkd], axis=1)
        t = np.sum(histograms["backgrounds"][bkd], axis=0)
        background_hists[bkd] = {"energy": e, "time": t}
    projected_histograms["backgrounds"] = background_hists

    return projected_histograms


def plot_observables(params: dict, histograms_1d_unosc: dict, histograms_1d_osc: dict, alpha) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]
    label_dict = {
        "brn": "BRNs",
        "nin": "NINs",
        "nuE": r"$\nu_e$",
        "nuMu": r"$\nu_\mu$",
        "nuTau": r"$\nu_\tau$",
        "nuEBar": r"$\bar{\nu}_e$",
        "nuMuBar": r"$\bar{\nu}_\mu$",
        "nuTauBar": r"$\bar{\nu}_\tau$",
        "nuS": r"$\nu_s$",
        "nuSBar": r"$\bar{\nu}_s$"
    }

    # stacked histogram
    x = []
    t = []
    e_weights = []
    t_weights = []
    labels = []

    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    # beam state subtraction
    pe_hist = (histograms_1d_unosc["beam_state"]["C"]["energy"] - histograms_1d_unosc["ssb"]["energy"] * ( 1 + alpha[3])) / np.diff(observable_bin_arr)
    t_hist = (histograms_1d_unosc["beam_state"]["C"]["time"] - histograms_1d_unosc["ssb"]["time"] * ( 1 + alpha[3])) / np.diff(t_bin_arr)
    pe_err = (np.sqrt(histograms_1d_unosc["beam_state"]["C"]["energy"] + histograms_1d_unosc["ssb"]["energy"] * ( 1 + alpha[3]))) / np.diff(observable_bin_arr)
    t_err = (np.sqrt(histograms_1d_unosc["beam_state"]["C"]["time"] + histograms_1d_unosc["ssb"]["time"] * ( 1 + alpha[3]))) / np.diff(t_bin_arr)

    # 3+1 model
    for bkd in histograms_1d_osc["backgrounds"].keys():
        if(bkd == "brn"): scale = alpha[1]
        if(bkd == "nin"): scale = alpha[2]
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(histograms_1d_osc["backgrounds"][bkd]["energy"] * (1 + scale)/np.diff(observable_bin_arr))
        t_weights.append(histograms_1d_osc["backgrounds"][bkd]["time"] * (1 + scale)/np.diff(t_bin_arr))

        labels.append(bkd)

    for flavor in histograms_1d_osc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        # don't plot empty flavors
        if np.sum(histograms_1d_osc["neutrinos"][flavor]["energy"]) == 0: continue
        if np.sum(histograms_1d_osc["neutrinos"][flavor]["time"]) == 0: continue
        scale = alpha[0]
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(histograms_1d_osc["neutrinos"][flavor]["energy"] * (1 + scale)/np.diff(observable_bin_arr))
        t_weights.append(histograms_1d_osc["neutrinos"][flavor]["time"] * (1 + scale)/np.diff(t_bin_arr))

        labels.append(flavor)

    # transform labels
    labels = [label_dict[label] for label in labels]

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

    for bkd in histograms_1d_unosc["backgrounds"].keys():
        if bkd == "brn": scale = alpha[1]
        if bkd == "nin": scale = alpha[2]
        e_weights += histograms_1d_unosc["backgrounds"][bkd]["energy"] * (1 + scale) / np.diff(observable_bin_arr)
        t_weights += histograms_1d_unosc["backgrounds"][bkd]["time"] * (1 + scale) / np.diff(t_bin_arr)

    for flavor in histograms_1d_unosc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        scale = alpha[0]
        e_weights += histograms_1d_unosc["neutrinos"][flavor]["energy"] * (1 + scale) / np.diff(observable_bin_arr)
        t_weights += histograms_1d_unosc["neutrinos"][flavor]["time"] * (1 + scale) / np.diff(t_bin_arr)

    ax[0].hist(observable_bin_arr[:-1], bins=observable_bin_arr, weights=e_weights, histtype='step', linestyle='dashed', label="No Oscillation", color="grey")
    ax[1].hist(t_bin_arr[:-1], bins=t_bin_arr, weights=t_weights, histtype='step', linestyle='dashed', label="No Oscillation", color="grey")

    # data
    ax[0].errorbar(x=(observable_bin_arr[1:] + observable_bin_arr[:-1])/2, y=pe_hist, ls="none", yerr=np.sqrt(pe_err), label="Data", color="black", marker="x", markersize=5)
    ax[1].errorbar(x=(t_bin_arr[1:] + t_bin_arr[:-1])/2, y=t_hist, ls="none", yerr=np.sqrt(t_err), label="Data", color="black", marker="x", markersize=5)
    
    ax[0].set_xlabel("Energy [PE]")
    ax[0].set_ylabel("Counts / PE")

    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_ylabel(r"Counts / $\mu$s")

    ax[1].set_xscale('function', functions=(lambda x: np.where(x < 1, x, (x - 1) / 3 + 1),
                                            lambda x: np.where(x < 1, x, 3 * (x - 1) + 1)))

    ax[0].legend(*map(reversed, ax[0].get_legend_handles_labels()))
    ax[1].legend(*map(reversed, ax[1].get_legend_handles_labels()))

    plt.plot()
    plt.show()

    return

def plot_observables2d(params: dict, histograms_unosc: dict, histograms_osc: dict, alpha) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    # x, y = np.meshgrid(observable_bin_arr[:-1], t_bin_arr[:-1])
    x = np.tile(observable_bin_arr[:-1], len(t_bin_arr[:-1]))
    y = np.repeat(t_bin_arr[:-1], len(observable_bin_arr[:-1]))

    # 3 + 1 model
    weights = 0

    for bkd in histograms_osc["backgrounds"].keys():
        if bkd == "brn": scale = alpha[1]
        if bkd == "nin": scale = alpha[2]
        weights += histograms_osc["backgrounds"][bkd] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))

    for flavor in histograms_osc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        scale = alpha[0]
        weights += histograms_osc["neutrinos"][flavor] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))

    ax[0, 0].set_title("3+1 Model Observables")
    ax[0,0].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="Greys")
    cbar = plt.colorbar(ax[0,0].collections[0], ax=ax[0,0])
    cbar.set_label(r"Counts / PE / $\mu$s")

    model_weights = weights

    # disappearance
    weights = 0
    for flavor in histograms_unosc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        scale = alpha[0]
        weights += histograms_unosc["neutrinos"][flavor] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))
                                                                                
    for flavor in histograms_osc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        scale = alpha[0]
        weights -= histograms_osc["neutrinos"][flavor] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))

    ax[0, 1].set_title("3+1 Model Steriles")
    ax[0,1].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="Greys")
    cbar = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1])
    cbar.set_label(r"Counts / PE / $\mu$s")

    # data residual
    weights = (histograms_unosc["beam_state"]["C"] - histograms_unosc["ssb"] * (1 + alpha[3])) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))
    weights -= model_weights
    ax[1, 0].set_title("Data Residual")
    ax[1,0].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="bwr", norm=colors.CenteredNorm())
    cbar = plt.colorbar(ax[1,0].collections[0], ax=ax[1,0])
    cbar.set_label(r"Counts / PE / $\mu$s")

    # stat likelihood
    predicted = 0
    for flavor in histograms_osc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        predicted += histograms_osc["neutrinos"][flavor] * (1 + alpha[0])
    predicted += histograms_osc["backgrounds"]["brn"] * (1 + alpha[1])
    predicted += histograms_osc["backgrounds"]["nin"] * (1 + alpha[2])
    predicted += histograms_osc["ssb"] * (1 + alpha[3])

    observed = histograms_unosc["beam_state"]["C"]

    weights = -predicted + observed * np.log(predicted) - gammaln(observed + 1)
    print(np.sum(weights))
    ax[1, 1].set_title("Log Likelihood")
    ax[1,1].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="Greys_r")
    cbar = plt.colorbar(ax[1,1].collections[0], ax=ax[1,1])
    cbar.set_label("Log Likelihood")

    # settings

    for a in ax.flat:
        a.set_yscale('function', functions=(lambda x: np.where(x < 1, x, (x - 1) / 3 + 1),
                                                lambda x: np.where(x < 1, x, 3 * (x - 1) + 1)))
        a.set_yticks(np.arange(0, 7.0, 1.0))
        a.set_yticks(np.arange(0, 1., 0.125), minor=True)
        a.tick_params(axis='y', direction='out')
        a.tick_params(axis='y', which='minor', direction='out')
        a.tick_params(axis='x', direction='out')
        a.set_xlabel("Energy [PE]")
        a.set_ylabel(r"Time [$\mu$s]")
    plt.plot()
    plt.show()