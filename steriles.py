
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import pandas as pd

from numpy import heaviside

import numpy as np

import uproot

from flux.nuflux import NeutrinoFlux

from flux.probabilities import Pab, sin2theta

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
    plot_fluxes = True
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


    if use_root:

        paper_rf = uproot.open("flux/snsFlux2D.root")
        convolved_energy_and_time_of_nu_mu = paper_rf["convolved_energy_time_of_nu_mu;1"]
        convolved_energy_and_time_of_nu_mu_bar = paper_rf["convolved_energy_time_of_anti_nu_mu;1"]
        convolved_energy_and_time_of_nu_e = paper_rf["convolved_energy_time_of_nu_e;1"]
        convolved_energy_and_time_of_nu_e_bar = paper_rf["convolved_energy_time_of_anti_nu_e;1"]

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

        # nu e bar
        NuEBar = convolved_energy_and_time_of_nu_e_bar.values()
        anc_keNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()
        anc_keNuEBar_values = np.sum(NuEBar, axis=0)
        anc_keNuEBar_values = anc_keNuEBar_values / np.sum(anc_keNuEBar_values) * 0.0001 / 0.0848
        anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
        anc_tNuEBar_values = np.sum(NuEBar, axis=1)
        anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)  * 0.0001 / 0.0848

    anc_energy_centered = (anc_keNuE_edges[1:] + anc_keNuE_edges[:-1]) / 2
    anc_time_centered = (anc_tNuE_edges[1:] + anc_tNuE_edges[:-1]) / 2

    L = 19.3
    deltam41 = 1
    Ue4 = 0.3162
    Umu4 = 0.1778/3
    Utau4 = 0.0
    oscillated_nu_e = Pab(anc_energy_centered, L, deltam41, 1, 1, Ue4, Umu4, Utau4)*anc_keNuE_values + Pab(anc_energy_centered, L, deltam41, 2, 1, Ue4, Umu4, Utau4)*anc_keNuMu_values
    oscillated_nu_e_t_ratio = np.sum(oscillated_nu_e) / np.sum(anc_keNuE_values)

    oscillated_nu_mu = Pab(anc_energy_centered, L, deltam41, 2, 2, Ue4, Umu4, Utau4)*anc_keNuMu_values + Pab(anc_energy_centered, L, deltam41, 1, 2, Ue4, Umu4, Utau4)*anc_keNuE_values
    oscillated_nu_mu_t_ratio = np.sum(oscillated_nu_mu) / np.sum(anc_keNuMu_values)

    oscillated_nu_mu_bar = Pab(anc_energy_centered, L, deltam41, 2, 2, Ue4, Umu4, Utau4)*anc_keNuMuBar_values + Pab(anc_energy_centered, L, deltam41, 1, 2, Ue4, Umu4, Utau4)*anc_keNuEBar_values
    oscillated_nu_mu_bar_t_ratio = np.sum(oscillated_nu_mu_bar) / np.sum(anc_keNuMuBar_values)

    oscillated_nu_e_bar = Pab(anc_energy_centered, L, deltam41, 1, 1, Ue4, Umu4, Utau4)*anc_keNuEBar_values +  Pab(anc_energy_centered, L, deltam41, 2, 1, Ue4, Umu4, Utau4)*anc_keNuMuBar_values
    oscillated_nu_e_bar_t_ratio = np.sum(oscillated_nu_e_bar) / np.sum(anc_keNuMuBar_values)

    if plot_fluxes == True: 
        # plot just the fluxes in energy and time
        fig, ax = plt.subplots(4, 2, figsize=(8, 6))
        ax = ax.flatten()

        #set figure title
        fig.suptitle(f"sin2(2theta_emu) = {sin2theta(1, 2, Ue4, Umu4, Utau4):.4f}, sin2(2theta_ee) = {sin2theta(1, 1, Ue4, Umu4, Utau4):.4f}")

        ax[0].step(anc_energy_centered, anc_keNuE_values, label="NuE")
        ax[0].step(anc_energy_centered, oscillated_nu_e, label="Oscillated NuE")
        ax[1].step(anc_tNuE_edges[1:], anc_tNuE_values, label="NuE")
        ax[1].step(anc_tNuE_edges[1:], oscillated_nu_e_t_ratio*anc_tNuE_values, label="Oscillated NuE")

        ax[2].step(anc_energy_centered, anc_keNuMu_values, label="NuMu")
        ax[2].step(anc_energy_centered, oscillated_nu_mu, label="Oscillated NuMu")
        ax[3].step(anc_tNuMu_edges[1:], anc_tNuMu_values, label="NuMu")
        ax[3].step(anc_tNuMu_edges[1:], oscillated_nu_mu_t_ratio*anc_tNuMu_values, label="Oscillated NuMu")

        ax[4].step(anc_energy_centered, anc_keNuMuBar_values, label="NuMuBar")
        ax[4].step(anc_energy_centered, oscillated_nu_mu_bar, label="Oscillated NuMuBar")
        ax[5].step(anc_tNuMuBar_edges[1:], anc_tNuMuBar_values, label="NuMuBar")
        ax[5].step(anc_tNuMuBar_edges[1:], oscillated_nu_mu_bar_t_ratio*anc_tNuMuBar_values, label="Oscillated NuMuBar")

        ax[6].step(anc_energy_centered, anc_keNuEBar_values, label="NuEBar")
        ax[6].step(anc_energy_centered, oscillated_nu_e_bar, label="Oscillated NuEBar")
        ax[7].step(anc_tNuEBar_edges[1:], anc_tNuEBar_values, label="NuEBar")
        ax[7].step(anc_tNuEBar_edges[1:], oscillated_nu_e_bar_t_ratio*anc_tNuMuBar_values, label="Oscillated NuEBar")

        for i in range(0, 8, 2):
            ax[i].set_xlabel("Energy (MeV)")
            ax[i].set_ylabel("Flux")
            ax[i].set_xlim(0, 60)
            ax[i].legend()
        
        for i in range(1, 8, 2):
            ax[i].set_xlabel("Time (ns)")
            ax[i].set_ylabel("Flux")
            ax[i].set_xlim(0, 6_000)
            ax[i].legend()

        plt.show()

        return
    
if __name__ == "__main__":
    main()