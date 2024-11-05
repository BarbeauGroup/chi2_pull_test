
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import pandas as pd

from numpy import heaviside

import numpy as np

import uproot

def csi_efficiency(x):
    # a = 0.6655
    # k = 0.4942
    # xzero = 10.8507

    # return (a / (1 + np.exp(-k*(x - xzero))))*heaviside(x - 5, 0.5)
    eps = (1.32045 / (1 + np.exp(-0.285979*(x - 10.8646)))) - 0.333322
    eps[eps < 0] = 0

    return eps

def csi_time_efficiency(t):
    a = 520
    b = 0.0494 / 1000

    if t < a:
        return 1
    elif t < 6000:
        return np.exp(-b*(t - a))
    else:
        return 0


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

def main():

    # flux = NeutrinoFlux("flux/snsFlux2D.root")

    # return 0

    # Debugging flags
    use_csi = True
    use_root = True
    use_nuEnergyAndTime = False
    plot_fluxes = False
    plot_efficiency = False
    use_paper_binning = True
    plot_final_histogram = True


    # CsI Detector Parameters and Normalizations
    if use_csi:
        mass = 14.6 # kg
        A = (132.91 + 126.90) / 2
        Da = 1.66e-27 # kg
        n_atoms = mass / (A * Da)
        number_of_neutrinos = (0.0848*3.198e23) / (4*np.pi*(19.3*100)**2)

        csi_flux_smearing_matrix = np.load("data/flux_transfer_matrices/csi_flux_smearing_matrix.npy")
        csi_quenching_detector_matrix = np.load("data/flux_transfer_matrices/csi_quenching_detector_matrix.npy")
        csi_energy_efficiency_matrix = np.load("data/flux_transfer_matrices/csi_energy_efficiency_matrix.npy")
        csi_time_efficiency_matrix = np.load("data/flux_transfer_matrices/csi_time_efficiency_matrix.npy")


    if use_root:

        paper_rf = uproot.open("flux/snsFlux2D.root")
        convolved_energy_and_time_of_nu_mu = paper_rf["convolved_energy_time_of_nu_mu;1"]
        convolved_energy_and_time_of_nu_mu_bar = paper_rf["convolved_energy_time_of_anti_nu_mu;1"]
        convolved_energy_and_time_of_nu_e = paper_rf["convolved_energy_time_of_nu_e;1"]

        # project onto x axis
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


    anc_energy_centered = (anc_keNuE_edges[1:] + anc_keNuE_edges[:-1]) / 2
    anc_time_centered = (anc_tNuE_edges[1:] + anc_tNuE_edges[:-1]) / 2

    if plot_fluxes == True: 
        # plot just the fluxes in energy and time
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].step(anc_energy_centered, anc_keNuE_values, label="NuE (paper)")
        ax[0].step(anc_energy_centered, anc_keNuMu_values, label="NuMu (paper)")
        ax[0].step(anc_energy_centered, anc_keNuMuBar_values, label="NuMuBar (paper)")

        ax[0].set_xlabel("Energy (MeV)")
        ax[0].set_ylabel("Flux")
        ax[0].set_ylim(0.04)
        ax[0].set_xlim(0, 60)
        ax[0].legend()

        ax[1].step(anc_tNuE_edges[1:], anc_tNuE_values, label="NuE (paper)")
        ax[1].step(anc_tNuMu_edges[1:], anc_tNuMu_values, label="NuMu (paper)")
        ax[1].step(anc_tNuMuBar_edges[1:], anc_tNuMuBar_values, label="NuMuBar (paper)")
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("Flux")
        ax[1].set_yscale("linear")
        ax[1].set_ylim(0.01)
        ax[1].set_xlim(0, 6_000)
        ax[1].legend()

        plt.show()

        return

    # This is the linear algebra step: flux -> true -> reconstructed


    nuMu_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuMu_values[1:60])
    nuMu_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMu_truth_level_energy)

    nuE_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuE_values[1:60])
    nuE_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuE_truth_level_energy)

    nuMuBar_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuMuBar_values[1:60])
    nuMuBar_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMuBar_truth_level_energy)

    observable_bin_arr = np.arange(0, len(nuE_observable_energy)/10, .1)

    nuE_true_counts = np.sum(nuE_observable_energy)*number_of_neutrinos*n_atoms
    nuMuBar_true_counts = np.sum(nuMuBar_observable_energy)*number_of_neutrinos*n_atoms
    nuMu_true_counts = np.sum(nuMu_observable_energy)*number_of_neutrinos*n_atoms

    nuE_integral = np.sum(csi_energy_efficiency_matrix*nuE_observable_energy)*number_of_neutrinos*n_atoms
    nuMuBar_integral = np.sum(csi_energy_efficiency_matrix*nuMuBar_observable_energy)*number_of_neutrinos*n_atoms
    nuMu_integral = np.sum(csi_energy_efficiency_matrix*nuMu_observable_energy)*number_of_neutrinos*n_atoms

    # Time-averaged efficiencies
    nuE_e_frac = np.sum(csi_energy_efficiency_matrix*nuE_observable_energy)/np.sum(nuE_observable_energy)
    nuMuBar_e_frac = np.sum(csi_energy_efficiency_matrix*nuMuBar_observable_energy)/np.sum(nuMuBar_observable_energy)
    nuMu_e_frac = np.sum(csi_energy_efficiency_matrix*nuMu_observable_energy)/np.sum(nuMu_observable_energy)

    if plot_efficiency == True:
        plt.plot(observable_bin_arr, csi_efficiency(observable_bin_arr))
        # put a tick every PE from 0 to 20
        plt.xticks(np.arange(0, 61))
        plt.xlim(0, 20)
        plt.show()

        return

    eps_t = np.ones_like(anc_time_centered)
    for i, t in enumerate(anc_time_centered):
        eps_t[i] = csi_time_efficiency(t)
    
    tNuE_frac = np.sum(csi_time_efficiency_matrix * anc_tNuE_values)
    tNuMuBar_frac = np.sum(csi_time_efficiency_matrix * anc_tNuMuBar_values)
    tNuMu_frac = np.sum(csi_time_efficiency_matrix * anc_tNuMu_values)

    print("t array shapes: ", anc_tNuE_values.shape, anc_tNuMuBar_values.shape, anc_tNuMu_values.shape, eps_t.shape)

    energy_cut = np.where(observable_bin_arr < 60)[0]
    eNuE_frac = np.sum(anc_keNuE_values[energy_cut])
    eNuMu_frac = np.sum(anc_keNuMu_values[energy_cut])
    eNuMuBar_frac = np.sum(anc_keNuMuBar_values[energy_cut])

    # print t fracs:
    print(f"tNuE_frac: {tNuE_frac}, tNuMuBar_frac: {tNuMuBar_frac}, tNuMu_frac: {tNuMu_frac}")

    nuE_counts = np.dot(csi_energy_efficiency_matrix, nuE_observable_energy)*number_of_neutrinos*n_atoms*tNuE_frac
    nuMuBar_counts = np.dot(csi_energy_efficiency_matrix, nuMuBar_observable_energy)*number_of_neutrinos*n_atoms*tNuMuBar_frac
    nuMu_counts = np.dot(csi_energy_efficiency_matrix,nuMu_observable_energy)*number_of_neutrinos*n_atoms*tNuMu_frac

    print("counts shapes: ", nuE_counts.shape, nuMuBar_counts.shape, nuMu_counts.shape)

    # ####################
    # # Load in BRN Data #
    # ####################
    brn_pe = np.loadtxt("data/csi_anc/brnPE.txt")
    brn_pe_bins_centers = brn_pe[:60,0]
    brn_pe_counts = brn_pe[:60,1]

    nin_pe = np.loadtxt("data/csi_anc/ninPE.txt")
    nin_pe_bins_centers = nin_pe[:60,0]
    nin_pe_counts = nin_pe[:60,1]

    bin_arr = np.ones(len(brn_pe_counts))*(brn_pe_bins_centers[1] - brn_pe_bins_centers[0])
    brn_pe_norm = np.sum(brn_pe_counts*bin_arr)
    print("brn pe raw counts: ", brn_pe_norm)
    bin_arr = np.ones(len(nin_pe_counts))*(nin_pe_bins_centers[1] - nin_pe_bins_centers[0])
    nin_pe_norm = np.sum(nin_pe_counts*bin_arr)
    print("nin pe raw counts: ", nin_pe_norm)

    rebinned_observable_bin_arr = np.asarray([0,8,12,16,20,24,32,40,50,60])
    rebin_weights_brn = np.ones(len(rebinned_observable_bin_arr) - 1)*(brn_pe_bins_centers[1] - brn_pe_bins_centers[0])/np.diff(rebinned_observable_bin_arr)
    rebin_weights_nin = np.ones(len(rebinned_observable_bin_arr) - 1)*(nin_pe_bins_centers[1] - nin_pe_bins_centers[0])/np.diff(rebinned_observable_bin_arr)

    rebinned_brn_counts = rebin_histogram(brn_pe_counts, brn_pe_bins_centers, rebinned_observable_bin_arr) * rebin_weights_brn * 18.4 / brn_pe_norm
    rebinned_nin_counts = rebin_histogram(nin_pe_counts, nin_pe_bins_centers, rebinned_observable_bin_arr) * rebin_weights_nin * 5.6 / nin_pe_norm

    rebinned_brn_nin_counts = rebinned_brn_counts + rebinned_nin_counts

    brn_t = np.loadtxt("data/csi_anc/brnTrec.txt")
    brn_t_bins_centers = brn_t[:,0]  
    brn_t_counts = brn_t[:,1]

    nin_t = np.loadtxt("data/csi_anc/ninTrec.txt")
    nin_t_bins_centers = nin_t[:,0]
    nin_t_counts = nin_t[:,1]

    bin_arr = np.ones(len(brn_t_counts))*(brn_t_bins_centers[1] - brn_t_bins_centers[0])
    brn_t_norm = np.sum(brn_t_counts*bin_arr)
    print("brn t raw counts: ", brn_t_norm)
    bin_arr = np.ones(len(nin_t_counts))*(nin_t_bins_centers[1] - nin_t_bins_centers[0])
    nin_t_norm = np.sum(nin_t_counts*bin_arr)
    print("nin t raw counts: ", nin_t_norm)

    t_binning = np.asarray([0, 1./8., 2./8., 3./8., 4./8.,  5./8., 6./8., 7./8., 1.0, 2.0, 4.0, 6.0])

    rebin_weights_brn = np.ones(len(t_binning) - 1)*(brn_t_bins_centers[1] - brn_t_bins_centers[0])/np.diff(t_binning)
    rebin_weights_nin = np.ones(len(t_binning) - 1)*(nin_t_bins_centers[1] - nin_t_bins_centers[0])/np.diff(t_binning)

    rebinned_brn_t_counts = rebin_histogram(brn_t_counts, brn_t_bins_centers, t_binning) * rebin_weights_brn * 18.4 / brn_t_norm
    rebinned_nin_t_counts = rebin_histogram(nin_t_counts, nin_t_bins_centers, t_binning) * rebin_weights_nin * 5.6 / nin_t_norm

    rebinned_brn_nin_t_counts = rebinned_brn_t_counts + rebinned_nin_t_counts

      

    # ###############################
    # # Load in the Ancillary Data  #
    # ###############################

    dataBeamOnAC = np.loadtxt("data/csi_anc/dataBeamOnAC.txt")
    AC_PE = dataBeamOnAC[:,0]
    AC_t = dataBeamOnAC[:,1]

    dataBeamOnC = np.loadtxt("data/csi_anc/dataBeamOnC.txt")
    C_PE = dataBeamOnC[:,0]
    C_t = dataBeamOnC[:,1]

    # must filter the events that have t > 6 
    AC_high_time_idx = np.where(AC_t > 6)[0]
    AC_PE = np.delete(AC_PE, AC_high_time_idx)
    AC_t = np.delete(AC_t, AC_high_time_idx)

    C_high_time_idx = np.where(C_t > 6)[0]
    C_PE = np.delete(C_PE, C_high_time_idx)
    C_t = np.delete(C_t, C_high_time_idx)

    # and energy > 60
    AC_high_energy_idx = np.where(AC_PE > 60)[0]
    AC_PE = np.delete(AC_PE, AC_high_energy_idx)
    AC_t = np.delete(AC_t, AC_high_energy_idx)

    C_high_energy_idx = np.where(C_PE > 60)[0]
    C_PE = np.delete(C_PE, C_high_energy_idx)
    C_t = np.delete(C_t, C_high_energy_idx)



    # ########################################
    # # Rebin the observable energy spectrum #
    # ########################################
    if use_paper_binning == True:

        # rebin the observable energy spectrum
        rebinned_observable_bin_arr = np.asarray([0,8,12,16,20,24,32,40,50,60])
        rebin_weights = np.ones(len(rebinned_observable_bin_arr) - 1)
        rebin_weights[0] = 1 / (rebinned_observable_bin_arr[1] -rebinned_observable_bin_arr[0])
        rebin_weights[1] = 1 / (rebinned_observable_bin_arr[2] -rebinned_observable_bin_arr[1])
        rebin_weights[2] = 1 / (rebinned_observable_bin_arr[3] -rebinned_observable_bin_arr[2])
        rebin_weights[3] = 1 / (rebinned_observable_bin_arr[4] -rebinned_observable_bin_arr[3])
        rebin_weights[4] = 1 / (rebinned_observable_bin_arr[5] -rebinned_observable_bin_arr[4])
        rebin_weights[5] = 1 / (rebinned_observable_bin_arr[6] -rebinned_observable_bin_arr[5])
        rebin_weights[6] = 1 / (rebinned_observable_bin_arr[7] -rebinned_observable_bin_arr[6])
        rebin_weights[7] = 1 / (rebinned_observable_bin_arr[8] -rebinned_observable_bin_arr[7])
        rebin_weights[8] = 1 / (rebinned_observable_bin_arr[9] -rebinned_observable_bin_arr[8])

        rebinned_nuE_counts = rebin_histogram(nuE_counts, observable_bin_arr, rebinned_observable_bin_arr) * rebin_weights
        rebinned_nuMuBar_counts = rebin_histogram(nuMuBar_counts, observable_bin_arr, rebinned_observable_bin_arr) * rebin_weights
        rebinned_nuMu_counts = rebin_histogram(nuMu_counts, observable_bin_arr, rebinned_observable_bin_arr) * rebin_weights


        # rebin the observable t spectrum
        rebin_weights = np.ones(len(t_binning) - 1)/(np.diff(t_binning))

        # Applied eps cut
        anc_tNuE_values_cut = np.dot(csi_time_efficiency_matrix, anc_tNuE_values) * nuE_true_counts * nuE_e_frac
        anc_tNuMuBar_values_cut = np.dot(csi_time_efficiency_matrix, anc_tNuMuBar_values) * nuMuBar_true_counts * nuMuBar_e_frac
        anc_tNuMu_values_cut = np.dot(csi_time_efficiency_matrix, anc_tNuMu_values) * nuMu_true_counts * nuMu_e_frac

        print("Total counts in energy plot ", np.sum(nuE_counts), np.sum(nuMuBar_counts), np.sum(nuMu_counts))
        print("Total counts in time plot ", np.sum(anc_tNuE_values_cut), np.sum(anc_tNuMuBar_values_cut), np.sum(anc_tNuMu_values_cut))

        rebinned_tnuE_counts = rebin_histogram(anc_tNuE_values_cut, anc_tNuE_edges/1000, t_binning) * rebin_weights
        rebinned_tnuMuBar_counts = rebin_histogram(anc_tNuMuBar_values_cut, anc_tNuMuBar_edges/1000, t_binning) * rebin_weights
        rebinned_tnuMu_counts = rebin_histogram(anc_tNuMu_values_cut, anc_tNuMu_edges/1000, t_binning) * rebin_weights

        print(np.sum(rebinned_nuE_counts), np.sum(rebinned_nuMuBar_counts), np.sum(rebinned_nuMu_counts))
        print(np.sum(rebinned_tnuE_counts), np.sum(rebinned_tnuMuBar_counts), np.sum(rebinned_tnuMu_counts))

    # print fractions
    print(f"Fractions: {tNuE_frac}, {tNuMuBar_frac}, {tNuMu_frac}, {eNuE_frac}, {eNuMuBar_frac}, {eNuMu_frac}")
    print(f"Events before cuts: {nuE_integral + nuMuBar_integral + nuMu_integral}")
    time_and_energy_cut = tNuE_frac*eNuE_frac + tNuMuBar_frac*eNuMuBar_frac + tNuMu_frac*eNuMu_frac # was multiplying by integral twice
    print(f"time and energy cut: {time_and_energy_cut}")

    print(rebinned_nuE_counts[1], rebinned_nuMuBar_counts[1], rebinned_nuMu_counts[1], rebinned_brn_nin_counts[1])
    print(rebinned_nuE_counts[4], rebinned_nuMuBar_counts[4], rebinned_nuMu_counts[4], rebinned_brn_nin_counts[4])


    # ####################################
    # # Bin the data in the same binning #
    # ####################################

    rebinned_observable_bin_arr = np.asarray([0,8,12,16,20,24,32,40,50,60])

    ac_pe_counts, ac_pe_edges = np.histogram(AC_PE, bins=rebinned_observable_bin_arr) 
    ac_pe_edges_centered = (ac_pe_edges[1:] + ac_pe_edges[:-1]) / 2
    ac_pe_counts = ac_pe_counts/np.diff(rebinned_observable_bin_arr)

    ac_t_counts, ac_t_edges = np.histogram(AC_t, bins=t_binning)
    ac_t_counts = ac_t_counts/np.diff(t_binning)
    ac_t_edges_centered = (ac_t_edges[1:] + ac_t_edges[:-1]) / 2

    c_pe_counts, c_pe_edges = np.histogram(C_PE, bins=rebinned_observable_bin_arr)
    c_pe_edges_centered = (c_pe_edges[1:] + c_pe_edges[:-1]) / 2
    c_pe_counts = c_pe_counts/np.diff(rebinned_observable_bin_arr)

    c_t_counts, c_t_edges = np.histogram(C_t, bins=t_binning)
    c_t_counts = c_t_counts/np.diff(t_binning)
    c_t_edges_centered = (c_t_edges[1:] + c_t_edges[:-1]) / 2

    plt.style.use(["science", "vibrant"])
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].errorbar(ac_pe_edges_centered, 
                      ac_pe_counts, 
                      yerr=np.sqrt(ac_pe_counts), 
                      xerr=np.diff(rebinned_observable_bin_arr)/2, 
                      fmt="o", 
                      markersize=0.5,
                      label="AC")
    ax[0].errorbar(c_pe_edges_centered,
                        c_pe_counts,
                        yerr=np.sqrt(c_pe_counts),
                        xerr=np.diff(rebinned_observable_bin_arr)/2,
                        fmt="o",
                        markersize=0.5,
                        label="C")
    
    ax[0].hist([rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1]], 
            
            bins=rebinned_observable_bin_arr, 

            weights=    [ rebinned_brn_nin_counts,
                        rebinned_nuMu_counts, 
                        rebinned_nuMuBar_counts, 
                        rebinned_nuE_counts],
        stacked=True, 
        label=['BRN+NIN', r'$\nu_\mu$', r'$\bar{\nu}_\mu$', r'$\nu_e$'],
        alpha=1,
        color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81"])
    
    ax[0].hist([rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1]], 
           bins=rebinned_observable_bin_arr, 
           weights=[rebinned_brn_nin_counts,
                    rebinned_nuMu_counts, 
                    rebinned_nuMuBar_counts, 
                    rebinned_nuE_counts],
           stacked=True, 
           histtype='step',
           edgecolor='black')
    

    ax[0].set_xlabel("Energy (PE)")
    ax[0].set_ylabel("Counts/PE")
    ax[0].set_xlim(0, 60)
    ax[0].set_ylim(0, 25)
    ax[0].legend()

    ax[1].errorbar(ac_t_edges_centered,
                        ac_t_counts,
                        yerr=np.sqrt(ac_t_counts),
                        xerr=np.diff(t_binning)/2,
                        fmt="o",
                        markersize=0.5,
                        label="AC")
    ax[1].errorbar(c_t_edges_centered,
                        c_t_counts,
                        yerr=np.sqrt(c_t_counts),
                        xerr=np.diff(t_binning)/2,
                        fmt="o",
                        markersize=0.5,
                        label="C")
    
    ax[1].hist([t_binning[:-1], t_binning[:-1], t_binning[:-1], t_binning[:-1]], 
            
            bins=t_binning, 

            weights=    [ rebinned_brn_nin_t_counts,
                        rebinned_tnuMu_counts,
                        rebinned_tnuMuBar_counts,
                        rebinned_tnuE_counts],
        stacked=True, 
        label=[r'BRN+NIN', r'$\nu_\mu$', r'$\bar{\nu}_\mu$', r'$\nu_e$'],
        alpha=1,
        color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81"])
    
    print("rebinned_brn_nin_t_counts: ", rebinned_brn_nin_t_counts)
    
    ax[1].hist([t_binning[:-1], t_binning[:-1], t_binning[:-1], t_binning[:-1]], 
           bins=t_binning, 
           weights=[rebinned_brn_nin_t_counts,
                    rebinned_tnuMu_counts,
                    rebinned_tnuMuBar_counts,
                    rebinned_tnuE_counts],
           stacked=True, 
           histtype='step',
           edgecolor='black')

    ax[1].set_xlabel(r"Time ($\mu$s)")
    ax[1].set_ylabel(r"Counts/$\mu$s")
    ax[1].set_xlim(0, 6)
    ax[1].set_ylim(0, 300)
    ax[1].legend()

    print("brn+nin counts ", np.sum(rebinned_brn_nin_counts*np.diff(rebinned_observable_bin_arr)))
    print("brn+nin t counts ", np.sum(rebinned_brn_nin_t_counts*np.diff(t_binning)))



    plt.show()
    return



    # ############################
    # # Plot the final histogram #
    # ############################

    if plot_final_histogram == True:
        # make a stacked histogram plot
        plt.style.use(["science", "vibrant"])
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].hist([rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1], rebinned_observable_bin_arr[:-1]], 
                    
                    bins=rebinned_observable_bin_arr, 

                    weights=    [ rebinned_brn_nin_counts,
                                rebinned_nuMu_counts, 
                                rebinned_nuMuBar_counts, 
                                rebinned_nuE_counts],
                stacked=True, 
                label=['BRN+NIN', r'$\nu_\mu$', r'$\bar{\nu}_\mu$', r'$\nu_e$'],
                alpha=0.75,
                color=["#009988", "#EE3377", "#EE7733", "#CC3311"])
        
        ax[0].annotate("Paper prediction: 341 +- 11 +- 42", xy=(0.5, 0.5), xycoords='axes fraction', fontsize=20)
        ax[0].annotate(f"I'm getting: {round(time_and_energy_cut,2)}", xy=(0.5, 0.45), xycoords='axes fraction', fontsize=20)
                

        # make the yticks font size larger
        # ax[0].set_yticks(fontsize=20)
        ax[0].legend(fontsize=20)
        ax[0].set_xlabel("Energy (PE)", fontsize=20) 
        ax[0].set_ylabel("Counts/PE", fontsize=20)

        ax[0].set_xlim(0, 60)
        ax[0].set_ylim(0, 25)

        # timing plot
        ax[1].plot(anc_time_centered, anc_tNuE_values, label="NuE (paper)")
        ax[1].plot(anc_time_centered, anc_tNuMu_values, label="NuMu (paper)")
        ax[1].plot(anc_time_centered, anc_tNuMuBar_values, label="NuMuBar (paper)")

        ax[1].set_xlim(0, 6_000)


        plt.show()
    


if __name__ == "__main__":
    main()
