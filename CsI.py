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

    # Debugging flags
    use_csi = True
    use_root = True
    use_nuEnergyAndTime = False
    plot_fluxes = False
    plot_efficiency = False


    # CsI Detector Parameters and Normalizations
    if use_csi:
        mass = 14.6 # kg
        A = (132.91 + 126.90) / 2
        Da = 1.66e-27 # kg
        n_atoms = mass / (A * Da)
        number_of_neutrinos = (0.0848*3.198e23) / (4*np.pi*(19.3*100)**2)

        csi_flux_smearing_matrix = np.load("data/flux_transfer_matrices/csi_flux_smearing_matrix.npy")
        csi_quenching_detector_matrix = np.load("data/flux_transfer_matrices/csi_quenching_detector_matrix.npy")


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
        ax[0].set_ylim(0, 0.04)
        ax[0].set_xlim(0, 60)
        ax[0].legend()

        ax[1].step(anc_tNuE_edges[1:], anc_tNuE_values, label="NuE (paper)")
        ax[1].step(anc_tNuMu_edges[1:], anc_tNuMu_values, label="NuMu (paper)")
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

    nuMu_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuMu_values[1:60])
    nuMu_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMu_truth_level_energy)

    nuE_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuE_values[1:60])
    nuE_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuE_truth_level_energy)

    nuMuBar_truth_level_energy = np.dot(csi_flux_smearing_matrix, anc_keNuMuBar_values[1:60])
    nuMuBar_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMuBar_truth_level_energy)

    observable_bin_arr = np.arange(0, len(nuE_observable_energy), 1)

    nuE_integral = np.sum(csi_efficiency(observable_bin_arr)*nuE_observable_energy)*number_of_neutrinos*n_atoms
    nuMuBar_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy)*number_of_neutrinos*n_atoms
    nuMu_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMu_observable_energy)*number_of_neutrinos*n_atoms

    print(f"Total number of events: {nuE_integral + nuMuBar_integral + nuMu_integral}")


    if plot_efficiency == True:
        plt.plot(observable_bin_arr, csi_efficiency(observable_bin_arr))
        # put a tick every PE from 0 to 20
        plt.xticks(np.arange(0, 60, 1))
        plt.xlim(0, 20)
        plt.show()

        return


    time_cut = np.where(anc_time_centered < 6000)[0]
    tNuE_frac = np.sum(anc_tNuE_values[time_cut])
    tNuMu_frac = np.sum(anc_tNuMu_values[time_cut])
    tNuMuBar_frac = np.sum(anc_tNuMuBar_values[time_cut])

    energy_cut = np.where(observable_bin_arr < 60)[0]
    eNuE_frac = np.sum(anc_keNuE_values[energy_cut])
    eNuMu_frac = np.sum(anc_keNuMu_values[energy_cut])
    eNuMuBar_frac = np.sum(anc_keNuMuBar_values[energy_cut])

    only_time_cut = tNuE_frac*nuE_integral + tNuMuBar_frac*nuMuBar_integral + tNuMu_frac*nuMu_integral
    time_and_energy_cut = tNuE_frac*nuE_integral*eNuE_frac + tNuMuBar_frac*nuMuBar_integral*eNuMuBar_frac + tNuMu_frac*nuMu_integral*eNuMu_frac

    print(f"only time cut: {only_time_cut}")
    print(f"time and energy cut: {time_and_energy_cut}")

    nuE_counts = csi_efficiency(observable_bin_arr)*nuE_observable_energy*number_of_neutrinos*n_atoms
    nuMuBar_counts = csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy*number_of_neutrinos*n_atoms
    nuMu_counts = csi_efficiency(observable_bin_arr)*nuMu_observable_energy*number_of_neutrinos*n_atoms


    # plt.plot(truth_level_energy, label = "Truth Level Energy Spectrum")
    # plt.step(observable_bin_arr, observable_energy, label = "Observable Energy Spectrum")
    # plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuE_observable_energy*number_of_neutrinos*n_atoms, label = "ve", stacked=True)
    # plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy*number_of_neutrinos*n_atoms, label = "vubar", stacked=True)
    # plt.step(observable_bin_arr, csi_efficiency(observable_bin_arr)*nuMu_observable_energy*number_of_neutrinos*n_atoms, label = "vu", stacked=True)

    # make a stacked histogram plot
    plt.style.use(["science", "muted"])
    plt.figure(figsize=(12, 6))
    plt.hist([observable_bin_arr[:-1], observable_bin_arr[:-1], observable_bin_arr[:-1]], bins=observable_bin_arr, 
             weights=[nuMu_counts[:-1], nuMuBar_counts[:-1], nuE_counts[:-1]], 
             stacked=True, 
             label=[r'$\nu_\mu$', r'$\bar{\nu}_\mu$', r'$\nu_e$'], 
             alpha=0.75,
             color=["#AA4499", "#DDCC77", "#CC6677"])
             

    # make the yticks font size larger
    plt.yticks(fontsize=20)
    # plt.ylim(0, 1.5*np.max(csi_efficiency(observable_bin_arr)*nuE_observable_energy))
    plt.legend(fontsize=20)
    plt.xlabel("Energy (PE)", fontsize=20) 
    plt.ylabel("Counts/PE", fontsize=20)
    plt.show()
    


if __name__ == "__main__":
    main()
