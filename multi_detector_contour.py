import numpy as np
from utils import P_nue_nue, chi2_stat, chi2_sys, chi2_total, toy_model
from experiment import Experiment
import matplotlib.pyplot as plt
import scienceplots

def main():
    # Define the toy values for this test
    E = 30  # MeV
    L1 = 20  # m
    L2 = 1.5*L1

    N_sm = 5e6

    N_sm_at_baseline1 = N_sm / (4 * np.pi * L1 ** 2)
    N_sm_at_baseline2 = N_sm / (4 * np.pi * L2 ** 2)

    N_observed1 = N_sm_at_baseline1 
    N_observed2 = N_sm_at_baseline2 

    print("N_sm_at_baseline", N_sm_at_baseline1)
    print("N_observed", N_observed1)

    print("\n")

    print("N_sm_at_baseline", N_sm_at_baseline2)
    print("N_observed", N_observed2)

    # Create an Experiment instance
    pb_glass_b1 = Experiment("pb_glass_b1")
    pb_glass_b1.set_distance(L1)
    pb_glass_b1.set_n_observed(N_observed1)
    pb_glass_b1.set_systematic_error_dict({"flux": 0.1})

    pb_glass_b2 = Experiment("pb_glass_b2")
    pb_glass_b2.set_distance(L2)
    pb_glass_b2.set_n_observed(N_observed2)
    pb_glass_b2.set_systematic_error_dict({"flux": 0.1})

    flux_norm_param_arr = np.linspace(-0.1, 0.1, 25)
    delta_m_squared_arr = np.linspace(0.1, 10, 300)
    sin_squared_arr = np.linspace(0.005, 0.5, 300)

    chi2_total_2d_arr1 = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr), len(sin_squared_arr)))
    chi2_total_2d_arr2 = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr), len(sin_squared_arr)))

    chi2_total_2d_arr_combined = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr), len(sin_squared_arr)))


    for i, flux_norm_param in enumerate(flux_norm_param_arr):
        for j, delta_m_squared in enumerate(delta_m_squared_arr):
            for k, sin_squared_2_theta_14 in enumerate(sin_squared_arr):
                chi2_total_2d_arr1[i, j, k] = chi2_total(pb_glass_b1, toy_model, [E, L1, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline1], 
                                                    [flux_norm_param], pb_glass_b1.get_systematic_error_dict())
                
                chi2_total_2d_arr2[i, j, k] = chi2_total(pb_glass_b2, toy_model, [E, L2, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline2],
                                                        [flux_norm_param], pb_glass_b2.get_systematic_error_dict())
                

                chi_stat1 = chi2_stat(pb_glass_b1, toy_model, [E, L1, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline1], [flux_norm_param])
                chi_stat2 = chi2_stat(pb_glass_b2, toy_model, [E, L2, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline2], [flux_norm_param])
                chi_sys = chi2_sys([flux_norm_param], pb_glass_b1.get_systematic_error_dict())

                chi2_total_2d_arr_combined[i, j, k] = chi_stat1 + chi_stat2 + chi_sys

            

    # Find the minimum chi2_total 
    # print("Minimum chi2_total", np.min(chi2_total_2d_arr_combined))
    # print("Index of minimum chi2_total", np.where(chi2_total_2d_arr_combined == np.min(chi2_total_2d_arr_combined)))

    min_index_1 = np.where(chi2_total_2d_arr1 == np.min(chi2_total_2d_arr1))
    min_index_2 = np.where(chi2_total_2d_arr2 == np.min(chi2_total_2d_arr2))
    min_indexes_combined = np.where(chi2_total_2d_arr_combined == np.min(chi2_total_2d_arr_combined))


    optimal_flux_index_1 = min_index_1[0][0]
    optimal_flux_index_2 = min_index_2[0][0]
    optimal_flux_index_combined = min_indexes_combined[0][0]

    chi2_total_2d_projection1 = chi2_total_2d_arr1[optimal_flux_index_1]
    chi2_total_2d_projection2 = chi2_total_2d_arr2[optimal_flux_index_2]
    chi2_2d_projection = chi2_total_2d_arr_combined[optimal_flux_index_combined]


    plt.style.use("science")
    plt.figure(figsize=(12, 10))
    # Plot the chi2_total contour
    # plt.contourf(delta_m_squared_arr, sin_squared_arr, chi2_total_2d_projection1.T, levels=[0.95,1.05], cmap="viridis")
    # plt.contourf(delta_m_squared_arr, sin_squared_arr, chi2_total_2d_projection2.T, levels=[0.95,1.05], cmap="bone")
    plt.contourf(sin_squared_arr, delta_m_squared_arr, chi2_2d_projection, levels=[0.95,1.05], cmap="plasma")
    # show minimum
    # plt.scatter(delta_m_squared_arr[min_indexes_combined[1]], sin_squared_arr[min_indexes_combined[2]], color="red", label="Minimum Chi2 Total")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$\sin^2(2\theta_{14})$")
    plt.ylabel(r"$\Delta m^2_{14}$")

    # plt.legend()


    # plt.legend()
    plt.savefig("chi2_exclusion.png")

if __name__ == "__main__":
    main()
