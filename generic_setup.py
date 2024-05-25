from utils.stats import chi2_stat, chi2_sys
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
    

    print(chi2_stat(experiments[0], 
              truth_level_prediction, 
              [experiments[0], 0, 0],
              [0]))
    
    print(chi2_stat(experiments[1],
                truth_level_prediction,
                [experiments[1], 0, 0],
                [0]))
    
        

    # Now let's minimize
    def f1(x):
        stat = chi2_stat(experiments[0], 
              truth_level_prediction, 
              [experiments[0], x[0], x[1]],
              [x[2]])
        
        sys = chi2_sys(experiments[0], [x[2]])

        return stat + sys
    
    def f2(x):
        stat = chi2_stat(experiments[1], 
              truth_level_prediction, 
              [experiments[1], x[0], x[1]],
              [x[2]])
        
        sys = chi2_sys(experiments[0], [x[2]])

        return stat + sys
    
    def f3(x):
        stat1 = chi2_stat(experiments[0],
                truth_level_prediction,
                [experiments[0], x[0], x[1]],
                [x[2]])
        
        stat2 = chi2_stat(experiments[1],
                truth_level_prediction,
                [experiments[1], x[0], x[1]],
                [x[2]])
        
        sys = chi2_sys(experiments[0], [x[2]])

        return stat1 + stat2 + sys
    
    res1 = minimize(f1, [0,0,0], method='nelder-mead', options={'xatol': 1e-6, 'disp': True})
    res2 = minimize(f2, [0,0,0], method='nelder-mead', options={'xatol': 1e-6, 'disp': True})
    res3 = minimize(f3, [0,0,0], method='nelder-mead', options={'xatol': 1e-6, 'disp': True})

    
    # Using the optimal parameters for the nuisance parameters,
    # Let's make our chi2 grid over the model parameters

    delta_m14_squared = np.linspace(.01, 10, 300)
    sin_squared_2_theta_14 = np.linspace(0.01, 1.0, 300)

    chi2_2d_arr_list = []

    for i in range(len(experiments)):
        chi2_2d_arr_list.append(np.zeros((len(delta_m14_squared), len(sin_squared_2_theta_14))))

    chi2_2d_arr = np.zeros((len(delta_m14_squared), len(sin_squared_2_theta_14)))
    predicted_counts_arr = np.zeros((len(delta_m14_squared), len(sin_squared_2_theta_14)))

    # debug stuff remove later
    osc_flux_arr = np.zeros((len(delta_m14_squared), len(sin_squared_2_theta_14)))
    energy_arr = np.arange(1, 55, 1)

    for i in range(len(delta_m14_squared)):
        for j in range(len(sin_squared_2_theta_14)):
            params1 = [delta_m14_squared[i], sin_squared_2_theta_14[j]]
            params1.append(res1.x[2:])

            params2 = [delta_m14_squared[i], sin_squared_2_theta_14[j]]
            params2.append(res2.x[2:])

            params3 = [delta_m14_squared[i], sin_squared_2_theta_14[j]]
            params3.append(res3.x[2:])



            for k in range(len(experiments)):
                if k == 0:
                    chi2_2d_arr_list[k][i,j] = f1(params1)
                if k == 1:
                    chi2_2d_arr_list[k][i,j] = f2(params2)

            chi2_2d_arr[i,j] = f3(params3)

            predicted_counts_arr[i,j] = truth_level_prediction(experiments[0], delta_m14_squared[i], sin_squared_2_theta_14[j])

            # debug stuff remove later
            osc_flux_val = sns_nue_spectrum(energy_arr) * P_nue_nue(energy_arr, experiments[0].get_distance(), delta_m14_squared[i], sin_squared_2_theta_14[j])
            osc_flux_arr[i,j] = np.sum(osc_flux_val)


    if(1):
        # Write to file for plots
        np.save("plots/data/delta_m14_squared.npy", delta_m14_squared)
        np.save("plots/data/sin_squared_2_theta_14.npy", sin_squared_2_theta_14)

        np.save("plots/data/chi2_2d_arr.npy", chi2_2d_arr)
        np.save("plots/data/predicted_counts_arr.npy", predicted_counts_arr)
        np.save("plots/data/osc_flux_arr.npy", osc_flux_arr)


    print("Optimal parameters: ", res1.x)
    print("Optimal parameters: ", res2.x)
    print("Optimal parameters: ", res3.x)
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

    plt.title("90\%CI, 1:10 S/B Ratio, Infinite Bkd Windows, 1 Standard SNS Year Each", fontsize=18)

    plt.plot(0, 0, "red", markersize=10, label="10,200kg at 20m")
    plt.plot(0, 0, "green", markersize=10, label="10,200kg at 30m")
    plt.plot(0,0, "gray", markersize=10, label="Combined")
    plt.fill(denton_sin_squared, denton_delta_m_squared, "orange", alpha=0.2, label=r"Denton 2022, 2$\sigma$")
    plt.plot(0.35, 1.25, "darkorange", marker="o", markersize=10, label="Denton uboone best fit")

    plt.fill(sage_sin_squared_pt1, sage_delta_m_squared_pt1, "purple", alpha=0.2, label=r"BEST 2022, 2$\sigma$")
    plt.fill(sage_sin_squared_pt2, sage_delta_m_squared_pt2, "purple", alpha=0.2)
    plt.plot(0.34, 1.25, "black", marker="o", markersize=10, label="BEST best fit")

    plt.xscale("log")
    plt.yscale("log")

    plt.contourf(sin_squared_2_theta_14, delta_m14_squared, chi2_2d_arr_list[0], levels = [4.45, 4.75], cmap="inferno")
    plt.contourf(sin_squared_2_theta_14, delta_m14_squared, chi2_2d_arr_list[1], levels = [4.45, 4.75], cmap="viridis")
    plt.contourf(sin_squared_2_theta_14, delta_m14_squared, chi2_2d_arr, levels = [4.45, 4.75], cmap="cividis")


    plt.xlabel(r"$\sin^2(2\theta_{14})$", fontsize=18)
    plt.ylabel(r"$\Delta m^2_{14}$", fontsize=18)

    plt.xlim(0.01, 1)
    plt.ylim(0.1, 10)

    plt.legend(fontsize=18)

    plt.show()

    


if __name__ == "__main__":
    main()
