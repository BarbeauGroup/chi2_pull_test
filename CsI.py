from utils.stats import chi2_stat, chi2_sys
from experiment import Experiment
from utils.predictions import truth_level_prediction, P_nue_nue, sns_nue_spectrum, sns_numubar_spectrum

import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import pandas as pd

from numpy import heaviside
from scipy.optimize import minimize

import numpy as np
import json

import uproot

def csi_efficiency(x):
    a = 0.6655
    k = 0.4942
    xzero = 10.8507

    # return (a / (1 + np.exp(-k*(x - xzero))))*heaviside(x - 5, 0.5)
    eps = (1.32045 / (1 + np.exp(-0.285979*(x - 10.8646)))) - 0.333322
    eps[eps < 0] = 0

    return eps



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

    use_root = True
    use_nuEnergyAndTime = False
    plot_fluxes = False

    use_csi = True

    # define energy array
    energy_arr = np.arange(1,60,1)

    # define time array 
    time_arr = np.linspace(0, 20_000, 161)
    # time_arr = 1000*np.asarray([0, 0.125, 0.25, .375, .5, .625, .75, .875, 1, 2, 4, 6, 20])

    # CsI Detector Parameters and Normalizations
    if use_csi:
        mass = 14.6 # kg
        A = (133 + 127) / 2
        Da = 1.66e-27 # kg
        n_atoms = mass / (A * Da)
        number_of_neutrinos = (0.0848*3.198e23) / (4*np.pi*(19.3*100)**2)

        csi_flux_smearing_matrix = np.load("data/flux_transfer_matrices/csi_flux_smearing_matrix.npy")
        csi_quenching_detector_matrix = np.load("data/flux_transfer_matrices/csi_quenching_detector_matrix.npy")


    # Define SNS Flux Analytically
    sns_nuMu_flux = np.zeros_like(energy_arr)
    sns_nuMu_flux[30] = 1
    sns_nuMuBar_flux = sns_numubar_spectrum(energy_arr) / np.sum(sns_numubar_spectrum(energy_arr))
    sns_nuE_flux = sns_nue_spectrum(energy_arr) / np.sum(sns_nue_spectrum(energy_arr))

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


        if use_nuEnergyAndTime:
            print("Using nuEnergyAndTime.root")

            # # read in SNS flux from root sim
            # rf = uproot.open("flux/nuEnergyAndTime.root")
            # keNuE = rf["keNuE"]
            # keNuE_bin_edges = keNuE.axis().edges()
            # keNuE_values = keNuE.values() 

            # keNuMu = rf["keNuMu"]
            # keNuMu_bin_edges = keNuMu.axis().edges()
            # keNuMu_values = keNuMu.values()

            # keNuMuBar = rf["keAntiNuMu"]
            # keNuMuBar_bin_edges = keNuMuBar.axis().edges()
            # keNuMuBar_values = keNuMuBar.values()

            # tNuE = rf["tNuE"]
            # tNuE_bin_edges = tNuE.axis().edges()
            # tNuE_values = tNuE.values()

            # tNuMu = rf["tNuMu"]
            # tNuMu_bin_edges = tNuMu.axis().edges()
            # tNuMu_values = tNuMu.values()

            # tNuMuBar = rf["tAntiNuMu"]
            # tNuMuBar_bin_edges = tNuMuBar.axis().edges()
            # tNuMuBar_values = tNuMuBar.values()


            # # Need to rebin the root sim to match the energy array
            # keNuE_rebinned_values = rebin_histogram(keNuE_values, keNuE_bin_edges, energy_arr)
            # keNuE_rebinned_values = keNuE_rebinned_values / np.sum(keNuE_rebinned_values)

            # keNuMu_rebinned_values = rebin_histogram(keNuMu_values, keNuMu_bin_edges, energy_arr)
            # keNuMu_rebinned_values = keNuMu_rebinned_values / np.sum(keNuMu_rebinned_values)

            # keNuMuBar_rebinned_values = rebin_histogram(keNuMuBar_values, keNuMuBar_bin_edges, energy_arr)
            # keNuMuBar_rebinned_values = keNuMuBar_rebinned_values / np.sum(keNuMuBar_rebinned_values)

            # tNuE_rebinned_values = rebin_histogram(tNuE_values, tNuE_bin_edges, time_arr)
            # tNuE_rebinned_values /= np.sum(tNuE_rebinned_values)

            # tNuMu_rebinned_values = rebin_histogram(tNuMu_values, tNuMu_bin_edges, time_arr)
            # tNuMu_rebinned_values /= np.sum(tNuMu_rebinned_values)

            # tNuMuBar_rebinned_values = rebin_histogram(tNuMuBar_values, tNuMuBar_bin_edges, time_arr)
            # tNuMuBar_rebinned_values /= np.sum(tNuMuBar_rebinned_values)





    anc_energy_centered = (anc_keNuE_edges[1:] + anc_keNuE_edges[:-1]) / 2
    anc_time_centered = (anc_tNuE_edges[1:] + anc_tNuE_edges[:-1]) / 2

    energy_arr_centered = (energy_arr[1:] + energy_arr[:-1]) / 2
    time_arr_centered = (time_arr[1:] + time_arr[:-1]) / 2



    if plot_fluxes == True: 
        # plot just the fluxes in energy and time
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        # ax[0].step(energy_arr_centered, keNuE_rebinned_values, label="NuE")
        ax[0].step(anc_energy_centered, anc_keNuE_values, label="NuE (paper)")

        # ax[0].step(energy_arr_centered, keNuMu_rebinned_values, label="NuMu")
        ax[0].step(anc_energy_centered, anc_keNuMu_values, label="NuMu (paper)")

        # ax[0].step(energy_arr_centered, keNuMuBar_rebinned_values, label="NuMuBar")
        ax[0].step(anc_energy_centered, anc_keNuMuBar_values, label="NuMuBar (paper)")

        ax[0].set_xlabel("Energy (MeV)")
        ax[0].set_ylabel("Flux")
        ax[0].set_ylim(0, 0.04)
        ax[0].set_xlim(0, 60)
        ax[0].legend()

        # ax[1].step(time_arr_centered, tNuE_rebinned_values, label="NuE (rebinned)")
        ax[1].step(anc_tNuE_edges[1:], anc_tNuE_values, label="NuE (paper)")

        # ax[1].step(time_arr_centered, tNuMu_rebinned_values, label="NuMu (rebinned)")
        ax[1].step(anc_tNuMu_edges[1:], anc_tNuMu_values, label="NuMu (paper)")

        # ax[1].step(time_arr_centered, tNuMuBar_rebinned_values, label="NuMuBar (rebinned)")
        ax[1].step(anc_tNuMuBar_edges[1:], anc_tNuMuBar_values, label="NuMuBar (paper)")
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("Flux")
        ax[1].set_yscale("linear")
        ax[1].set_ylim(0, 0.01)
        ax[1].set_xlim(0, 6_000)
        ax[1].legend()

        plt.show()

        return

    # This is the linear algebra step: flux -> true -> reconstructed

    nuMu_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuMu_values[:59])
    nuMu_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMu_truth_level_energy)

    nuE_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuE_values[:59])
    nuE_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuE_truth_level_energy)

    nuMuBar_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuMuBar_values[:59])
    nuMuBar_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMuBar_truth_level_energy)

    observable_bin_arr = np.arange(0, len(nuE_observable_energy), 1)

    nuE_integral = np.sum(csi_efficiency(observable_bin_arr)*nuE_observable_energy)*number_of_neutrinos*n_atoms
    nuMuBar_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy)*number_of_neutrinos*n_atoms
    nuMu_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMu_observable_energy)*number_of_neutrinos*n_atoms

    print(f"Total number of events: {nuE_integral + nuMuBar_integral + nuMu_integral}")


    time_cut = np.where(anc_time_centered < 6000)[0]
    tNuE_frac = np.sum(anc_tNuE_values[time_cut])
    tNuMu_frac = np.sum(anc_tNuMu_values[time_cut])
    tNuMuBar_frac = np.sum(anc_tNuMuBar_values[time_cut])

    print(f"Total number of events (time cut): {tNuE_frac*nuE_integral + tNuMuBar_frac*nuMuBar_integral + tNuMu_frac*nuMu_integral}")

    return


    # plt.plot(truth_level_energy, label = "Truth Level Energy Spectrum")
    # plt.step(observable_bin_arr, observable_energy, label = "Observable Energy Spectrum")
    plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuE_observable_energy, label = "ve")
    plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy, label = "vmubar")
    plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuMu_observable_energy, label = "vmu")

    plt.annotate(f"ve: {round(nuE_integral,2)}", (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=16)
    plt.annotate(f"vubar: {round(nuMuBar_integral,2)}", (0, 0), (90, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=16)
    plt.annotate(f"vu: {round(nuMu_integral,2)}", (0, 0), (210, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=16)
    # make the yticks font size larger
    plt.yticks(fontsize=20)
    plt.ylim(0, 1.5*np.max(csi_efficiency(observable_bin_arr)*nuE_observable_energy))
    plt.legend()
    plt.show()





    


if __name__ == "__main__":
    main()
