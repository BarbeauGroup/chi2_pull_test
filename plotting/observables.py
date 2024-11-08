import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from utils.histograms import rebin_histogram

plt.style.use(['science', 'high-contrast'])

def plot_observables(unosc: dict, osc: dict, bkd_dict: dict) -> None:

    # TODO: Make this a parameter
    rebinned_observable_bin_arr = np.asarray([0,8,12,16,20,24,32,40,50,60])
    rebin_weights = np.ones(len(rebinned_observable_bin_arr) - 1) / np.diff(rebinned_observable_bin_arr)

    rebinned_t_bin_arr = np.asarray([0, 1./8., 2./8., 3./8., 4./8.,  5./8., 6./8., 7./8., 1.0, 2.0, 4.0, 6.0])
    rebin_t_weights = np.ones(len(rebinned_t_bin_arr) - 1) / np.diff(rebinned_t_bin_arr)



    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    x = []
    t = []
    e_weights = []
    t_weights = []
    labels = []

    for bkd in bkd_dict.keys():
        if bkd == "brn": norm = 18.4
        else: norm = 5.6
        x.append(rebinned_observable_bin_arr[:-1])
        t.append(rebinned_t_bin_arr[:-1])
        e_weights.append(norm*rebin_histogram(bkd_dict[bkd]["energy"][1], bkd_dict[bkd]["energy"][0], rebinned_observable_bin_arr)*rebin_weights)
        t_weights.append(norm*rebin_histogram(bkd_dict[bkd]["time"][1], bkd_dict[bkd]["time"][0], rebinned_t_bin_arr)*rebin_t_weights)

        labels.append(bkd)

    for flavor in unosc.keys():

        x.append(rebinned_observable_bin_arr[:-1])
        t.append(rebinned_t_bin_arr[:-1])
        e_weights.append(rebin_histogram(unosc[flavor]["energy"][1], unosc[flavor]["energy"][0], rebinned_observable_bin_arr)*rebin_weights)
        t_weights.append(rebin_histogram(unosc[flavor]["time"][1], unosc[flavor]["time"][0]/1000, rebinned_t_bin_arr)*rebin_t_weights)

        labels.append(flavor)
    
    ax[0].hist(x, bins=rebinned_observable_bin_arr, weights=e_weights,
                stacked=True, 
                histtype='step',
                edgecolor='black')
    
    ax[0].hist(x, bins=rebinned_observable_bin_arr, weights=e_weights,
            stacked=True, 
            label=labels,
            alpha=1,
            # color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81", "green", "black", "blue", "red"])
            color=["black", "red", "sienna", "darkorange", "gold", "greenyellow", "seagreen", "mediumturquoise", "dodgerblue", "mediumorchid"])            
    
    ax[0].set_xlabel("Energy [PE]")
    ax[0].set_ylabel("Counts / PE")

    ax[1].hist(t, bins=rebinned_t_bin_arr, weights=t_weights,
                   stacked=True, 
                    histtype='step',
                    edgecolor='black')

    ax[1].hist(t, bins=rebinned_t_bin_arr, weights=t_weights,
                stacked=True, 
                label=labels,
                alpha=1,
                # color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81", "green", "black", "blue", "red"])
                color=["black", "red", "sienna", "darkorange", "gold", "greenyellow", "seagreen", "mediumturquoise", "dodgerblue", "mediumorchid"])
    
    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_ylabel(r"Counts / $\mu$s")


    e_weights = 0
    t_weights = 0
    for flavor in osc.keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        # e_weights += osc[flavor]["energy"][1]
        # t_weights += osc[flavor]["time"][1]
        e_weights += rebin_histogram(e_weights, osc[flavor]["energy"][0], rebinned_observable_bin_arr)*rebin_weights
        t_weights += rebin_histogram(t_weights, osc[flavor]["time"][0]/1000, rebinned_t_bin_arr)*rebin_t_weights

    for bkd in bkd_dict.keys():
        if bkd == "brn": norm = 18.4
        else: norm = 5.6
        x.append(rebinned_observable_bin_arr[:-1])
        t.append(rebinned_t_bin_arr[:-1])
        e_weights += norm*rebin_histogram(bkd_dict[bkd]["energy"][1], bkd_dict[bkd]["energy"][0], rebinned_observable_bin_arr)*rebin_weights
        t_weights += norm*rebin_histogram(bkd_dict[bkd]["time"][1], bkd_dict[bkd]["time"][0], rebinned_t_bin_arr)*rebin_t_weights
    
    ax[0].errorbar(x=(rebinned_observable_bin_arr[1:] + rebinned_observable_bin_arr[:-1])/2,
            #    y=rebin_histogram(e_weights, osc[flavor]["energy"][0], rebinned_observable_bin_arr)*rebin_weights,
                y=e_weights,
               xerr=np.diff(rebinned_observable_bin_arr)/2, 
               label="Oscillated", 
               color="blue", ls="none", 
               marker="x", markersize=5)
    
    ax[1].errorbar(x=(rebinned_t_bin_arr[1:] + rebinned_t_bin_arr[:-1])/2,
                # y=rebin_histogram(t_weights, osc[flavor]["time"][0]/1000, rebinned_t_bin_arr)*rebin_t_weights,
                y=t_weights,
                xerr=np.diff(rebinned_t_bin_arr)/2, 
                label="Oscillated", 
                color="blue", ls="none", 
                marker="x", markersize=5)

    ax[0].legend()
    ax[1].legend()

    plt.plot()
    plt.show()

    return