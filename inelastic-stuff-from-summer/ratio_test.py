from utils.stats import chi2_stat_ratio
from experiment import Experiment
from utils.predictions import truth_level_prediction, sns_nue_spectrum, P_nue_nue

import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import pandas as pd

from scipy.optimize import minimize

import numpy as np
import json

def main():

    with open("config/pb_glasses_20m_30m_1y_each.json", "r") as f:
        config = json.load(f)

    detector_dict = config["Detectors"]

    # Make an array of experiment objects
    experiments = []
    for i,detector in enumerate(detector_dict):
        # experiments.append(Experiment(detector_dict[detector]))
        experiments.append(Experiment(detector))

        experiments[i].set_distance(detector_dict[detector]["Distance"])
        experiments[i].set_n_observed(detector_dict[detector]["N observed"])
        experiments[i].set_mass(detector_dict[detector]["Mass"])
        experiments[i].set_exposure(detector_dict[detector]["Exposure"])
        experiments[i].set_steady_state_background(detector_dict[detector]["Steady state background"])
        experiments[i].set_number_background_windows(detector_dict[detector]["Number of background windows"])
        experiments[i].set_systematic_error_dict(detector_dict[detector]["Systematic uncertainties"])
        experiments[i].set_flux_transfer_matrix(np.load(detector_dict[detector]["Flux Transfer Matrix"]))

    
    # Let's do a verbose call of truth_level_prediction
    truth_level_prediction(experiments[0], 0, 0, verbose = True)
    truth_level_prediction(experiments[1], 0, 0, verbose = True)

    
    # print("Nominal ratio: ", 1/ (experiments[0].distance**2 / experiments[1].distance**2))
    # print("chi2 example: ", chi2_stat_ratio(experiments[0], experiments[1], truth_level_prediction,[0,0]))

    # return 0

    delta_m14_squared = np.linspace(.01, 50, 500)
    sin_squared_2_theta_14 = np.linspace(0.01, 1.0, 500)

    chi2_2d_arr = np.zeros((len(delta_m14_squared), len(sin_squared_2_theta_14)))


    for i in range(len(delta_m14_squared)):
        for j in range(len(sin_squared_2_theta_14)):
            
            chi2_2d_arr[i, j] = chi2_stat_ratio(experiments[0], experiments[1], truth_level_prediction, [delta_m14_squared[i], sin_squared_2_theta_14[j]])




    if(1): 
        # Load up the data thief contours
        # Load data thief data
        data1 = np.loadtxt("data/bad_data_thief/denton_uboone.csv", delimiter=",", skiprows=1)
        denton_sin_squared = data1[:, 0]
        denton_delta_m_squared = data1[:, 1]

        data2 = np.loadtxt("data/bad_data_thief/best_sage_gallex_pt1.csv", delimiter=",", skiprows=1)
        sage_sin_squared_pt1 = data2[:, 0]
        sage_delta_m_squared_pt1 = data2[:, 1]

        data3 = np.loadtxt("data/bad_data_thief/best_sage_gallex_pt2.csv", delimiter=",", skiprows=1)
        sage_sin_squared_pt2 = data3[:, 0]
        sage_delta_m_squared_pt2 = data3[:, 1]


    # Plot the chi2_total contour
    plt.style.use("science")
    plt.figure(figsize=(8, 6))

    plt.title("90\%CI, 1:10 S/B Ratio, Infinite Bkd Windows, 1yr 20m, 1yr 30m", fontsize=18)

    # plt.plot(0, 0, "red", markersize=10, label="10,200kg at 20m")
    # plt.plot(0, 0, "green", markersize=10, label="10,200kg at 30m")
    # plt.plot(0,0, "gray", markersize=10, label="Combined")

    if(1):
        plt.fill(denton_sin_squared, denton_delta_m_squared, "orange", alpha=0.2, label=r"Denton 2022, 2$\sigma$")
        plt.plot(0.35, 1.25, "darkorange", marker="o", markersize=10, label="Denton uboone best fit")

        plt.fill(sage_sin_squared_pt1, sage_delta_m_squared_pt1, "purple", alpha=0.2, label=r"BEST 2022, 2$\sigma$")
        plt.fill(sage_sin_squared_pt2, sage_delta_m_squared_pt2, "purple", alpha=0.2)
        plt.plot(0.34, 1.25, "black", marker="o", markersize=10, label="BEST best fit")

    plt.xscale("log")
    plt.yscale("log")

    # plt.contourf(sin_squared_2_theta_14, delta_m14_squared, chi2_2d_arr_list[0], levels = [0, 6.18], cmap="inferno", alpha=0.5)
    plt.contourf(sin_squared_2_theta_14, delta_m14_squared, chi2_2d_arr, levels = [4.5, 4.7], cmap="inferno", alpha=0.5)


    plt.xlabel(r"$\sin^2(2\theta_{14})$", fontsize=18)
    plt.ylabel(r"$\Delta m^2_{14}$", fontsize=18)

    plt.xlim(0.005, 1)
    plt.ylim(0.1, 100)

    plt.legend(fontsize=18)

    plt.show()

    


if __name__ == "__main__":
    main()
