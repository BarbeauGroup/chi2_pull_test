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

    mass = 14.6 # kg
    A = (133 + 127) / 2
    Da = 1.66e-27 # kg
    n_atoms = mass / (A * Da)

    number_of_neutrinos = (0.08*3.2e23) / (4*np.pi*(19.3*100)**2)

    csi_flux_smearing_matrix = np.load("data/flux_transfer_matrices/csi_flux_smearing_matrix.npy")
    csi_quenching_detector_matrix = np.load("data/flux_transfer_matrices/csi_quenching_detector_matrix.npy")

    energy_arr = np.arange(1,60,1)

    # Define SNS Flux Analytically
    sns_nuMu_flux = np.zeros_like(energy_arr)
    sns_nuMu_flux[30] = 1
    sns_nuMuBar_flux = sns_numubar_spectrum(energy_arr) / np.sum(sns_numubar_spectrum(energy_arr))
    sns_nuE_flux = sns_nue_spectrum(energy_arr) / np.sum(sns_nue_spectrum(energy_arr))

    # read in SNS flux from root sim
    rf = uproot.open("flux/nuEnergyAndTime.root")
    keNuE = rf["keNuE"]
    keNuE_bin_edges = keNuE.axis().edges()
    keNuE_values = keNuE.values() 
    keNuE_values = keNuE_values

    

    # Need to rebin the root sim to match the energy array
    keNuE_rebinned_values = rebin_histogram(keNuE_values, keNuE_bin_edges, energy_arr)
    keNuE_rebinned_values = keNuE_rebinned_values / np.sum(keNuE_rebinned_values)

    energy_arr_centered = (energy_arr[1:] + energy_arr[:-1]) / 2


    # plot just the fluxes
    plt.figure(figsize=(10,7))
    plt.step(energy_arr, sns_nuE_flux, label = "ve")
    plt.step(energy_arr_centered, keNuE_rebinned_values, label = "ve (root)")
    # plt.step(energy_arr, sns_nuMuBar_flux, label = "vu")
    # plt.step(energy_arr, sns_nuMu_flux, label = "vuBar")
    plt.yscale('linear')
    plt.legend()
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
