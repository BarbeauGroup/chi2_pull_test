import numpy as np
from utils.utils import P_nue_nue, chi2_stat, chi2_sys, chi2_total, toy_model
from experiment import Experiment
import matplotlib.pyplot as plt
import scienceplots

def main():
    # Define the toy values for this test
    E = 30  # MeV
    L1 = 20  # m
    L2 = 70

    N_sm = 5e6

    N_sm_at_baseline1 = 3*785*0.689*.9 # N_sm / (4 * np.pi * L1 ** 2)
    N_sm_at_baseline2 = 3*785*0.689*.9 # N_sm / (4 * np.pi * L2 ** 2)

    N_observed1 = N_sm_at_baseline1 
    N_observed2 = N_sm_at_baseline2 

    print("N_sm_at_baseline", N_sm_at_baseline1)
    print("N_observed", N_observed1)
    print("")

    print("\n")

    print("N_sm_at_baseline", N_sm_at_baseline2)
    print("N_observed", N_observed2)

    # Create an Experiment instance
    pb_glass_b1_bkd_free = Experiment("pb_glass_b1_bkd_free")
    pb_glass_b1_bkd_free.set_distance(L1)
    pb_glass_b1_bkd_free.set_n_observed(N_observed1)
    pb_glass_b1_bkd_free.set_steady_state_background(0)
    pb_glass_b1_bkd_free.set_number_background_windows(1)
    pb_glass_b1_bkd_free.set_systematic_error_dict({"flux": 0.1})

    pb_glass_b2_bkd_free = Experiment("pb_glass_b2_bkd_free")
    pb_glass_b2_bkd_free.set_distance(L2)
    pb_glass_b2_bkd_free.set_n_observed(N_observed2)
    pb_glass_b2_bkd_free.set_steady_state_background(0)
    pb_glass_b2_bkd_free.set_number_background_windows(1)
    pb_glass_b2_bkd_free.set_systematic_error_dict({"flux": 0.1})

    pb_glass_b1_11 = Experiment("pb_glass_b1_one_to_one_ratio")
    pb_glass_b1_11.set_distance(L1)
    pb_glass_b1_11.set_n_observed(N_observed1)
    pb_glass_b1_11.set_steady_state_background(1)
    pb_glass_b1_11.set_number_background_windows(1)
    pb_glass_b1_11.set_systematic_error_dict({"flux": 0.1})


    pb_glass_b2_11 = Experiment("pb_glass_b2_one_to_one_ratio")
    pb_glass_b2_11.set_distance(L2)
    pb_glass_b2_11.set_n_observed(N_observed2)
    pb_glass_b2_11.set_steady_state_background(1)
    pb_glass_b2_11.set_number_background_windows(1)
    pb_glass_b2_11.set_systematic_error_dict({"flux": 0.1})


    pb_glass_b1_101 = Experiment("pb_glass_b1_ten_to_one_ratio")
    pb_glass_b1_101.set_distance(L1)
    pb_glass_b1_101.set_n_observed(N_observed1)
    pb_glass_b1_101.set_steady_state_background(10)
    pb_glass_b1_101.set_number_background_windows(1)
    pb_glass_b1_101.set_systematic_error_dict({"flux": 0.1})

    pb_glass_b2_101 = Experiment("pb_glass_b2_ten_to_one_ratio")
    pb_glass_b2_101.set_distance(L2)
    pb_glass_b2_101.set_n_observed(N_observed2)
    pb_glass_b2_101.set_steady_state_background(10)
    pb_glass_b2_101.set_number_background_windows(1)
    pb_glass_b2_101.set_systematic_error_dict({"flux": 0.1})



    flux_norm_param_arr = np.linspace(-0.1, 0.1, 10)
    delta_m_squared_arr = np.linspace(0.1, 10, 300)
    sin_squared_arr = np.linspace(0.01, 0.5, 300)


    chi2_total_arr_combined_bkd_free = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr), len(sin_squared_arr)))
    chi2_total_arr_combined_11 = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr), len(sin_squared_arr)))
    chi2_total_arr_combined_101 = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr), len(sin_squared_arr)))


    for i, flux_norm_param in enumerate(flux_norm_param_arr):
        for j, delta_m_squared in enumerate(delta_m_squared_arr):
            for k, sin_squared_2_theta_14 in enumerate(sin_squared_arr):
                
                # Background Free

                chi_stat1_bkd_free = chi2_stat(pb_glass_b1_bkd_free, toy_model, [E, L1, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline1], [flux_norm_param])
                chi_stat2_bkd_free = chi2_stat(pb_glass_b2_bkd_free, toy_model, [E, L2, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline2], [flux_norm_param])
                chi_sys_bkd_free = chi2_sys([flux_norm_param], pb_glass_b1_bkd_free.get_systematic_error_dict())

                chi2_total_arr_combined_bkd_free[i, j, k] = chi_stat1_bkd_free + chi_stat2_bkd_free + chi_sys_bkd_free

                # 1:1 Background Ratio

                chi_stat1_11 = chi2_stat(pb_glass_b1_11, toy_model, [E, L1, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline1], [flux_norm_param])
                chi_stat2_11 = chi2_stat(pb_glass_b2_11, toy_model, [E, L2, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline2], [flux_norm_param])
                chi_sys_11 = chi2_sys([flux_norm_param], pb_glass_b1_11.get_systematic_error_dict())

                chi2_total_arr_combined_11[i, j, k] = chi_stat1_11 + chi_stat2_11 + chi_sys_11

                # 10:1 Background Ratio

                chi_stat1_101 = chi2_stat(pb_glass_b1_101, toy_model, [E, L1, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline1], [flux_norm_param])
                chi_stat2_101 = chi2_stat(pb_glass_b2_101, toy_model, [E, L2, delta_m_squared, sin_squared_2_theta_14, N_sm_at_baseline2], [flux_norm_param])
                chi_sys_101 = chi2_sys([flux_norm_param], pb_glass_b1_101.get_systematic_error_dict())

                chi2_total_arr_combined_101[i, j, k] = chi_stat1_101 + chi_stat2_101 + chi_sys_101

            



    min_indexes_combined_bkd_free = np.where(chi2_total_arr_combined_bkd_free == np.min(chi2_total_arr_combined_bkd_free))
    optimal_flux_index_combined_bkd_free = min_indexes_combined_bkd_free[0][0]
    chi2_2d_projection_bkd_free = chi2_total_arr_combined_bkd_free[optimal_flux_index_combined_bkd_free]

    min_indexes_combined_11 = np.where(chi2_total_arr_combined_11 == np.min(chi2_total_arr_combined_11))
    optimal_flux_index_combined_11 = min_indexes_combined_11[0][0]
    chi2_2d_projection_11 = chi2_total_arr_combined_11[optimal_flux_index_combined_11]

    min_indexes_combined_101 = np.where(chi2_total_arr_combined_101 == np.min(chi2_total_arr_combined_101))
    optimal_flux_index_combined_101 = min_indexes_combined_101[0][0]
    chi2_2d_projection_101 = chi2_total_arr_combined_101[optimal_flux_index_combined_101]


    # Load data thief
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


    # plt.contourf(sin_squared_arr, delta_m_squared_arr, chi2_2d_projection_bkd_free, levels=[5.9,6.1], cmap="plasma")
    # plt.contourf(sin_squared_arr, delta_m_squared_arr, chi2_2d_projection_11, levels=[5.9,6.1], cmap="viridis")
    plt.contourf(sin_squared_arr, delta_m_squared_arr, chi2_2d_projection_101, levels=[5.9,6.1], cmap="cividis")

    plt.fill(denton_sin_squared, denton_delta_m_squared, "orange", alpha=0.2, label=r"Denton 2022, 2$\sigma$")
    plt.plot(0.35, 1.25, "darkorange", marker="o", markersize=10, label="Denton uboone best fit")

    plt.fill(sage_sin_squared_pt1, sage_delta_m_squared_pt1, "purple", alpha=0.2, label=r"BEST 2022, 2$\sigma$")
    plt.fill(sage_sin_squared_pt2, sage_delta_m_squared_pt2, "purple", alpha=0.2)
    plt.plot(0.34, 1.25, "black", marker="o", markersize=10, label="BEST best fit")

    # at coordinates 0.01, 0.1 make an annotation that will act as a legend
    # because contourf does not have labels
    # so plot a small horizontal red line for bkd free, green for 1:1, and blue for 10:1
    # plt.plot(0, 0, "red", markersize=10, label="Background Free")
    # plt.plot(0, 0, "green", markersize=10, label="1:1 Background Ratio")
    plt.plot(0, 0, "gray", markersize=10, label="1:10 S/B Ratio")




    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$\sin^2(2\theta_{14})$", fontsize=18)
    plt.ylabel(r"$\Delta m^2_{14}$", fontsize=18)

    plt.title("20t Pb Glass at Two Baselines (Still Buggy)", fontsize=15)
    plt.xlim(0.01, 0.5)
    plt.ylim(0.1, 10)

    plt.legend(fontsize=18)


    # plt.legend()
    plt.savefig("chi2_exclusion.png")

if __name__ == "__main__":
    main()
