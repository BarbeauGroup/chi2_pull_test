import matplotlib.pyplot as plt
import scienceplots
import numpy as np


def rebin_histogram(counts, bin_edges, new_bin_edges):

    new_counts = np.zeros(len(new_bin_edges) - 1)

    for i in range(len(new_bin_edges) - 1):
        new_bin_start, new_bin_end = new_bin_edges[i], new_bin_edges[i+1]
        
        # Loop over each fine bin
        for j in range(len(bin_edges) - 1):
            fine_bin_start, fine_bin_end = bin_edges[j], bin_edges[j+1]
            
            # Check for overlap between fine bin and new bin
            overlap_start = max(new_bin_start, fine_bin_start)
            overlap_end = min(new_bin_end, fine_bin_end)
            
            if overlap_start < overlap_end:
                # Calculate the overlap width
                overlap_width = overlap_end - overlap_start
                fine_bin_width = fine_bin_end - fine_bin_start
                
                # Proportion of the fine bin's count that goes into the new bin
                contribution = (overlap_width / fine_bin_width) * counts[j]
                
                # Add the contribution to the new bin's count
                new_counts[i] += contribution
    
    return new_counts


plt.style.use(['science', 'high-contrast'])

def plot_observables(unosc: dict, osc: dict) -> None:

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
            color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81", "green", "black", "blue", "red"])
    
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
                color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81", "green", "black", "blue", "red"])
    
    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_ylabel(r"Counts / $\mu$s")


    e_weights = 0
    t_weights = 0
    for flavor in osc.keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        e_weights += osc[flavor]["energy"][1]
        t_weights += osc[flavor]["time"][1]
    
    ax[0].errorbar(x=(rebinned_observable_bin_arr[1:] + rebinned_observable_bin_arr[:-1])/2,
               y=rebin_histogram(e_weights, osc[flavor]["energy"][0], rebinned_observable_bin_arr)*rebin_weights,
               xerr=np.diff(rebinned_observable_bin_arr)/2, 
               label="Oscillated", 
               color="blue", ls="none", 
               marker="x", markersize=5)
    
    ax[1].errorbar(x=(rebinned_t_bin_arr[1:] + rebinned_t_bin_arr[:-1])/2,
                y=rebin_histogram(t_weights, osc[flavor]["time"][0]/1000, rebinned_t_bin_arr)*rebin_t_weights,
                xerr=np.diff(rebinned_t_bin_arr)/2, 
                label="Oscillated", 
                color="blue", ls="none", 
                marker="x", markersize=5)

    ax[0].legend()
    ax[1].legend()

    plt.plot()
    plt.show()

    return