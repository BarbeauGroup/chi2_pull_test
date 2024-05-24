from utils.utils import P_nue_nue, chi2_stat, chi2_sys, chi2_total, toy_model
from experiment import Experiment

import matplotlib.pyplot as plt
import scienceplots

import numpy as np
import json

def main():

    with open("config/pb_glasses.json", "r") as f:
        config = json.load(f)

    detector_dict = config["Detectors"]

    # Make an array of experiment objects
    experiments = []
    for i,detector in enumerate(detector_dict):
        experiments.append(Experiment(detector_dict[detector]))
        experiments[i].set_distance(detector_dict[detector]["Distance"])
        experiments[i].set_n_observed(detector_dict[detector]["N observed"])
        experiments[i].set_steady_state_background(detector_dict[detector]["Steady state background"])
        experiments[i].set_number_background_windows(detector_dict[detector]["Number of background windows"])
        experiments[i].set_systematic_error_dict(detector_dict[detector]["Systematic uncertainties"])

    
    # Let's do some flux transfer matrix stuff........
    # And then move it to a function in utils.py
    m_mu = 105 # MeV
    def sns_nue_spectrum(E):
        f = 96 * E**2 * m_mu**-4 * (m_mu - 2*E)
        f[f < 0] = 0
        return f 
        
    knu_array =  np.arange(1, 55, 1)
    flux_arr = sns_nue_spectrum(knu_array)
    

    # Now we need to load the flux transfer matrix
    flux_transfer_matrix = np.load("data/flux_transfer_matrices/pb_CC.npy")
    flux_transfer_matrix = flux_transfer_matrix[:len(knu_array), :len(knu_array)]


    # Now we need to calculate the expected number of events
    em_spectrum = np.dot(flux_transfer_matrix, flux_arr)
    em_spectrum /= np.sum(em_spectrum)

    plt.plot(knu_array,flux_arr, label="knu", drawstyle="steps")
    plt.plot(knu_array, em_spectrum, label="em", drawstyle="steps")
    plt.legend()
    plt.show()








if __name__ == "__main__":
    main()
