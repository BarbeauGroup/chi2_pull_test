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


def csi_efficiency(x):
    a = 0.6655
    k = 0.4942
    xzero = 10.8507

    return (a / (1 + np.exp(-k*(x - xzero))))*heaviside(x - 5, 0.5)

def main():

    mass = 14.6 # kg
    A = (133 + 127) / 2
    Da = 1.66e-27 # kg
    n_atoms = mass / (A * Da)

    number_of_neutrinos = (0.08*3.2e23) / (4*np.pi*(19.3*100)**2)

    csi_flux_smearing_matrix = np.load("data/flux_transfer_matrices/csi_flux_smearing_matrix.npy")
    csi_quenching_detector_matrix = np.load("data/flux_transfer_matrices/csi_quenching_detector_matrix.npy")

    energy_arr = np.arange(1,60,1)

    sns_nuMu_flux = np.zeros_like(energy_arr)
    sns_nuMu_flux[30] = 1
    nuMu_truth_level_energy = np.dot(csi_flux_smearing_matrix, sns_nuMu_flux)
    nuMu_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMu_truth_level_energy)


    sns_nuE_flux = sns_nue_spectrum(energy_arr) / np.sum(sns_nue_spectrum(energy_arr))
    nuE_truth_level_energy = np.dot(csi_flux_smearing_matrix, sns_nuE_flux)
    nuE_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuE_truth_level_energy)

    sns_nuMuBar_flux = sns_numubar_spectrum(energy_arr) / np.sum(sns_numubar_spectrum(energy_arr))
    nuMuBar_truth_level_energy = np.dot(csi_flux_smearing_matrix, sns_nuMuBar_flux)
    nuMuBar_observable_energy = np.dot(np.nan_to_num(csi_quenching_detector_matrix), nuMuBar_truth_level_energy)


    observable_bin_arr = np.arange(0, len(nuE_observable_energy), 1)

    nuE_integral = np.sum(csi_efficiency(observable_bin_arr)*nuE_observable_energy)*number_of_neutrinos*n_atoms
    nuMuBar_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMuBar_observable_energy)*number_of_neutrinos*n_atoms
    nuMu_integral = np.sum(csi_efficiency(observable_bin_arr)*nuMu_observable_energy)*number_of_neutrinos*n_atoms

    # # plot just the fluxes
    # plt.figure(figsize=(10,7))
    # plt.plot(energy_arr, sns_nuE_flux, label = "ve")
    # plt.plot(energy_arr, sns_nuMuBar_flux, label = "vu")
    # plt.legend()
    # plt.show()

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
