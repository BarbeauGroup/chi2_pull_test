
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import pandas as pd

from numpy import heaviside

import numpy as np

import uproot

from flux.probabilities import Pab, sin2theta
from flux.nuflux import read_flux_from_root

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



    flux = read_flux_from_root("flux/snsFlux2D.root")
    
    print("Total number of muon neutrinos: ", np.sum(flux['NuMuEnergy'][1]))
    print("Total number of muon anti-neutrinos: ", np.sum(flux['NuMuBarEnergy'][1]))
    print("\n")
    print("Total number of electron neutrinos: ", np.sum(flux['NuEEnergy'][1]))
    print("Total number of electron anti-neutrinos: ", np.sum(flux['NuEBarEnergy'][1]))
    print("\n")
    print("Total number of tau neutrinos: ", np.sum(flux['NuTauEnergy'][1]))
    print("Total number of tau anti-neutrinos: ", np.sum(flux['NuTauBarEnergy'][1]))


    # Sterile Osc Parameters
    L = 19.3
    deltam41 = 1
    Ue4 = 0.3162
    Umu4 = 0.1778/3
    Utau4 = 0.0

    # Oscillation
    oscillated_nu_e = Pab(flux["NuEEnergy"][0], L, deltam41, 1, 1, Ue4, Umu4, Utau4)*flux["NuEEnergy"][1] + Pab(flux["NuMuEnergy"][0], L, deltam41, 2, 1, Ue4, Umu4, Utau4)*flux["NuEEnergy"][1]
    oscillated_nu_e_t_ratio = np.sum(oscillated_nu_e) / np.sum(flux["NuEEnergy"][1])

    oscillated_nu_mu = Pab(flux["NuMuEnergy"][0], L, deltam41, 2, 2, Ue4, Umu4, Utau4)*flux["NuMuEnergy"][1] + Pab(flux["NuEEnergy"][0], L, deltam41, 1, 2, Ue4, Umu4, Utau4)*flux["NuMuEnergy"][1]
    oscillated_nu_mu_t_ratio = np.sum(oscillated_nu_mu) / np.sum(flux["NuMuEnergy"][1])

    oscillated_nu_mu_bar = Pab(flux["NuMuBarEnergy"][0], L, deltam41, 2, 2, Ue4, Umu4, Utau4)*flux["NuMuBarEnergy"][1] + Pab(flux["NuEEnergy"][0], L, deltam41, 1, 2, Ue4, Umu4, Utau4)*flux["NuMuBarEnergy"][1]
    oscillated_nu_mu_bar_t_ratio = np.sum(oscillated_nu_mu_bar) / np.sum(flux["NuMuBarEnergy"][1])

    oscillated_nu_e_bar = Pab(flux["NuEEnergy"][0], L, deltam41, 1, 1, Ue4, Umu4, Utau4)*flux["NuEBarEnergy"][1] +  Pab(flux["NuMuBarEnergy"][0], L, deltam41, 2, 1, Ue4, Umu4, Utau4)*flux["NuEBarEnergy"][1]
    oscillated_nu_e_bar_t_ratio = np.sum(oscillated_nu_e_bar) / np.sum(flux["NuEBarEnergy"][1])


    print("Total number of oscillated muon neutrinos: ", np.sum(oscillated_nu_mu))
    print("Total number of oscillated muon anti-neutrinos: ", np.sum(oscillated_nu_mu_bar))
    print("\n")
    print("Total number of oscillated electron neutrinos: ", np.sum(oscillated_nu_e))
    print("Total number of oscillated electron anti-neutrinos: ", np.sum(oscillated_nu_e_bar))
    print("\n")
    print("Total number of tau neutrinos: ", np.sum(flux['NuTauEnergy'][1]))
    print("Total number of tau anti-neutrinos: ", np.sum(flux['NuTauBarEnergy'][1]))

    return

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