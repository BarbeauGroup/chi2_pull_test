import numpy as np
from utils import P_nue_nue, chi2_stat, chi2_sys, chi2_total, toy_model
from experiment import Experiment
import matplotlib.pyplot as plt

def main():
    # Define the toy values for this test
    E = 30  # MeV
    L1 = 20  # m
    L2 = 1.5*L1

    true_delta_m14_squared = 3
    true_sin_squared_2_theta_14 = 0.5

    N_sm = 1e6

    N_sm_at_baseline1 = N_sm / (4 * np.pi * L1 ** 2)
    N_sm_at_baseline2 = N_sm / (4 * np.pi * L2 ** 2)

    N_observed1 = N_sm_at_baseline1 * P_nue_nue(E, L1, true_delta_m14_squared, true_sin_squared_2_theta_14)
    N_observed2 = N_sm_at_baseline2 * P_nue_nue(E, L2, true_delta_m14_squared, true_sin_squared_2_theta_14)

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


    flux_norm_param_arr = np.linspace(-0.5, 0.5, 600)
    delta_m_squared_arr = np.linspace(1, 4, 600)

    chi2_total_2d_arr1 = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr)))
    chi2_total_2d_arr2 = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr)))

    chi2_total_2d_arr_combined = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr)))



    for i, flux_norm_param in enumerate(flux_norm_param_arr):
        for j, delta_m_squared in enumerate(delta_m_squared_arr):
            chi2_total_2d_arr1[i, j] = chi2_total(pb_glass_b1, toy_model, [E, L1, delta_m_squared, true_sin_squared_2_theta_14, N_sm_at_baseline1], 
                                                 [flux_norm_param], pb_glass_b1.get_systematic_error_dict())
            
            chi2_total_2d_arr2[i, j] = chi2_total(pb_glass_b2, toy_model, [E, L2, delta_m_squared, true_sin_squared_2_theta_14, N_sm_at_baseline2],
                                                    [flux_norm_param], pb_glass_b2.get_systematic_error_dict())
            

            chi_stat1 = chi2_stat(pb_glass_b1, toy_model, [E, L1, delta_m_squared, true_sin_squared_2_theta_14, N_sm_at_baseline1], [flux_norm_param])
            chi_stat2 = chi2_stat(pb_glass_b2, toy_model, [E, L2, delta_m_squared, true_sin_squared_2_theta_14, N_sm_at_baseline2], [flux_norm_param])
            chi_sys = chi2_sys([flux_norm_param], pb_glass_b1.get_systematic_error_dict())

            chi2_total_2d_arr_combined[i, j] = chi_stat1 + chi_stat2 + chi_sys

            

    # Find the minimum chi2_total 
    # min_indexes = np.where(chi2_total_2d_arr == np.min(chi2_total_2d_arr))
    # print("Minimum chi2_total", np.min(chi2_total_2d_arr))
    # print("Minimum chi2_total indexes", min_indexes)

    # Contour plot of chi2_total
    plt.contourf(flux_norm_param_arr, delta_m_squared_arr, chi2_total_2d_arr1.T, levels=[0.95,1.05], cmap="plasma")
    plt.contourf(flux_norm_param_arr, delta_m_squared_arr, chi2_total_2d_arr2.T, levels=[0.95,1.05], cmap="viridis")
    plt.contourf(flux_norm_param_arr, delta_m_squared_arr, chi2_total_2d_arr_combined.T, levels=[0.95,1.05], cmap="inferno")

    # plt.scatter(0, true_delta_m14_squared, color="red", label="True Value")
    # show minimum
    # plt.scatter(flux_norm_param_arr[min_indexes[0]], delta_m_squared_arr[min_indexes[1]], color="green", label="Min Value")
    plt.xlabel("flux_norm_param")
    plt.ylabel("delta_m_squared")
    plt.title("Chi2 Total Contour")

    plt.xlim(-0.2,0.2)
    plt.ylim(2.5,3.5)

    # plt.legend()
    plt.savefig("chi2_total_contour.png")

if __name__ == "__main__":
    main()
