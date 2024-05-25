import numpy as np
from utils.utils import P_nue_nue, chi2_stat, chi2_sys, chi2_total, toy_model
from experiment import Experiment
import matplotlib.pyplot as plt

def main():
    # Define the toy values for this test
    E = 30  # MeV
    L = 20  # m

    true_delta_m14_squared = 3
    true_sin_squared_2_theta_14 = 0.5

    N_sm = 5e6

    N_sm_at_baseline = N_sm / (4 * np.pi * L ** 2)
    N_observed = N_sm_at_baseline * P_nue_nue(E, L, true_delta_m14_squared, true_sin_squared_2_theta_14)

    print("N_sm_at_baseline", N_sm_at_baseline)
    print("N_observed", N_observed)

    # Create an Experiment instance
    pb_glass_b1 = Experiment("pb_glass_b1")
    pb_glass_b1.set_distance(L)
    pb_glass_b1.set_n_observed(N_observed)
    pb_glass_b1.set_systematic_error_dict({"flux": 0.1})

    flux_norm_param_arr = np.linspace(-0.5, 0.5, 600)
    delta_m_squared_arr = np.linspace(1, 4, 600)

    chi2_total_2d_arr = np.zeros((len(flux_norm_param_arr), len(delta_m_squared_arr)))

    for i, flux_norm_param in enumerate(flux_norm_param_arr):
        for j, delta_m_squared in enumerate(delta_m_squared_arr):
            chi2_total_2d_arr[i, j] = chi2_total(pb_glass_b1, toy_model, [E, L, delta_m_squared, true_sin_squared_2_theta_14, N_sm_at_baseline], 
                                                 [flux_norm_param], pb_glass_b1.get_systematic_error_dict())
            

    # Find the minimum chi2_total 
    min_indexes = np.where(chi2_total_2d_arr == np.min(chi2_total_2d_arr))
    print("Minimum chi2_total", np.min(chi2_total_2d_arr))
    print("Minimum chi2_total indexes", min_indexes)

    # Contour plot of chi2_total
    # plt.contourf(flux_norm_param_arr, delta_m_squared_arr, chi2_total_2d_arr, levels=np.linspace(0, 20, 10), cmap="viridis")
    plt.contourf(flux_norm_param_arr, delta_m_squared_arr, chi2_total_2d_arr.T, levels=[0.95,1.05], cmap="plasma")
    # plt.contourf(flux_norm_param_arr, delta_m_squared_arr, chi2_total_2d_arr.T, levels=[3.95,4.05], cmap="viridis")

    plt.scatter(0, true_delta_m14_squared, color="red", label="True Value")
    # show minimum
    plt.scatter(flux_norm_param_arr[min_indexes[0]], delta_m_squared_arr[min_indexes[1]], color="green", label="Min Value")
    plt.xlabel("flux_norm_param")
    plt.ylabel("delta_m_squared")
    plt.title("Chi2 Total Contour")

    plt.xlim(-0.2,0.2)
    plt.ylim(2.5,3.5)

    plt.legend()
    plt.savefig("chi2_total_contour.png")

if __name__ == "__main__":
    main()
