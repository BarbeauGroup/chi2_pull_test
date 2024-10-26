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

    return (a / (1 + np.exp(-k*(x - xzero))))*heaviside(x - 5, 0.5)


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

    # define energy array
    energy_arr = np.arange(1,60,1)

    # define time array 
    time_arr = np.linspace(0, 20_000, 161)

    mass = 14.6 # kg
    A = (133 + 127) / 2
    Da = 1.66e-27 # kg
    n_atoms = mass / (A * Da)

    number_of_neutrinos = (0.08*3.2e23) / (4*np.pi*(19.3*100)**2)

    csi_flux_smearing_matrix = np.load("data/flux_transfer_matrices/csi_flux_smearing_matrix.npy")
    csi_quenching_detector_matrix = np.load("data/flux_transfer_matrices/csi_quenching_detector_matrix.npy")


    # Define SNS Flux Analytically
    sns_nuMu_flux = np.zeros_like(energy_arr)
    sns_nuMu_flux[30] = 1
    sns_nuMuBar_flux = sns_numubar_spectrum(energy_arr) / np.sum(sns_numubar_spectrum(energy_arr))
    sns_nuE_flux = sns_nue_spectrum(energy_arr) / np.sum(sns_nue_spectrum(energy_arr))

    if use_root:
        # read in SNS flux from root sim
        rf = uproot.open("flux/nuEnergyAndTime.root")
        keNuE = rf["keNuE"]
        keNuE_bin_edges = keNuE.axis().edges()
        keNuE_values = keNuE.values() 

        keNuMu = rf["keNuMu"]
        keNuMu_bin_edges = keNuMu.axis().edges()
        keNuMu_values = keNuMu.values()

        keNuMuBar = rf["keAntiNuMu"]
        keNuMuBar_bin_edges = keNuMuBar.axis().edges()
        keNuMuBar_values = keNuMuBar.values()

        tNuE = rf["tNuE"]
        tNuE_bin_edges = tNuE.axis().edges()
        tNuE_values = tNuE.values()

        tNuMu = rf["tNuMu"]
        tNuMu_bin_edges = tNuMu.axis().edges()
        tNuMu_values = tNuMu.values()

        tNuMuBar = rf["tAntiNuMu"]
        tNuMuBar_bin_edges = tNuMuBar.axis().edges()
        tNuMuBar_values = tNuMuBar.values()


        # Need to rebin the root sim to match the energy array
        keNuE_rebinned_values = rebin_histogram(keNuE_values, keNuE_bin_edges, energy_arr)
        keNuE_rebinned_values = keNuE_rebinned_values / np.sum(keNuE_rebinned_values)

        keNuMu_rebinned_values = rebin_histogram(keNuMu_values, keNuMu_bin_edges, energy_arr)
        keNuMu_rebinned_values = keNuMu_rebinned_values / np.sum(keNuMu_rebinned_values)

        keNuMuBar_rebinned_values = rebin_histogram(keNuMuBar_values, keNuMuBar_bin_edges, energy_arr)
        keNuMuBar_rebinned_values = keNuMuBar_rebinned_values / np.sum(keNuMuBar_rebinned_values)

        tNuE_rebinned_values = rebin_histogram(tNuE_values, tNuE_bin_edges, time_arr)
        tNuE_rebinned_values /= np.sum(tNuE_rebinned_values)

        tNuMu_rebinned_values = rebin_histogram(tNuMu_values, tNuMu_bin_edges, time_arr)
        tNuMu_rebinned_values /= np.sum(tNuMu_rebinned_values)

        tNuMuBar_rebinned_values = rebin_histogram(tNuMuBar_values, tNuMuBar_bin_edges, time_arr)
        tNuMuBar_rebinned_values /= np.sum(tNuMuBar_rebinned_values)



    energy_arr_centered = (energy_arr[1:] + energy_arr[:-1]) / 2
    time_arr_centered = (time_arr[1:] + time_arr[:-1]) / 2


    # plot just the fluxes in energy and time
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].step(energy_arr_centered, keNuE_rebinned_values, label="NuE")
    ax[0].step(energy_arr_centered, keNuMu_rebinned_values, label="NuMu")
    ax[0].step(energy_arr_centered, keNuMuBar_rebinned_values, label="NuMuBar")
    ax[0].set_xlabel("Energy (MeV)")
    ax[0].set_ylabel("Flux")
    ax[0].set_ylim(0, 0.04)
    ax[0].legend()

    ax[1].step(time_arr_centered, tNuE_rebinned_values, label="NuE (rebinned)")
    ax[1].step(time_arr_centered, tNuMu_rebinned_values, label="NuMu (rebinned)")
    ax[1].step(time_arr_centered, tNuMuBar_rebinned_values, label="NuMuBar (rebinned)")
    ax[1].set_xlabel("Time (ns)")
    ax[1].set_ylabel("Flux")
    ax[1].set_yscale("linear")
    ax[1].set_ylim(0, 0.06)
    ax[1].set_xlim(0, 6_000)
    ax[1].legend()

    plt.show()

    return



    nuMu_truth_level_energy = np.dot(csi_flux_smearing_matrix, sns_nuMu_flux)
    nuMu_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMu_truth_level_energy)

    nuE_truth_level_energy = np.dot(csi_flux_smearing_matrix, sns_nuE_flux)
    nuE_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuE_truth_level_energy)

    nuMuBar_truth_level_energy = np.dot(csi_flux_smearing_matrix, sns_nuMuBar_flux)
    nuMuBar_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMuBar_truth_level_energy)


    observable_bin_arr = np.arange(0, len(nuE_observable_energy), 1)

    nuE_integral = np.sum(csi_efficiency(observable_bin_arr)*nuE_observable_energy)*number_of_neutrinos*n_atoms
    nuMuBar_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy)*number_of_neutrinos*n_atoms
    nuMu_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMu_observable_energy)*number_of_neutrinos*n_atoms



    # plt.plot(truth_level_energy, label = "Truth Level Energy Spectrum")
    # plt.step(observable_bin_arr, observable_energy, label = "Observable Energy Spectrum")
    plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuE_observable_energy, label = "ve")
    plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy, label = "vmubar")
    plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuMu_observable_energy, label = "vmu")

    plt.annotate(f"Number of events: {round(nuE_integral,2)}", (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=20)
    plt.annotate(f"Number of events: {round(nuMuBar_integral,2)}", (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=20)
    plt.annotate(f"Number of events: {round(nuMu_integral,2)}", (0, 0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=20)
    # make the yticks font size larger
    plt.yticks(fontsize=20)
    plt.legend()
    plt.show()





    


if __name__ == "__main__":
    main()
