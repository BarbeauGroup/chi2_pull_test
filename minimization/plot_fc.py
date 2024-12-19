import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the data
    grid_mins = np.load("output/real_csi2_ee_chi2_fc.npy")
    grid_samples = np.load("output/real_csi2_ee_costs_fc.npy")

    # Bins
    sin_bins = np.logspace(-3, 0, num=5, endpoint=True)
    mass_bins = np.logspace(0, 2, num=5)

    alpha = 0.9

    # Global minimum
    chi2_g_min = np.min(grid_mins)

    acceptance = np.full_like(grid_mins, False)
    c = np.full_like(grid_mins, 0.0)

    # Plot chi^2 dist for one bin
    c_alpha = np.percentile(grid_samples[4, 4], 100 * alpha)
    plt.hist(grid_samples[4, 4] -  chi2_g_min, bins=30)
    plt.axvline(c_alpha - chi2_g_min, color='r', label=f"c_alpha = {c_alpha - chi2_g_min}")
    plt.xlabel("Chi^2")
    plt.ylabel("Frequency")
    plt.show()
    return

    for i in range(len(sin_bins)):
        for j in range(len(mass_bins)):
            # Check if the minimum is within the confidence interval
            # Calculate c_alpha
            c_alpha = np.percentile(grid_samples[i, j], 100 * alpha)
            chi2_i_min = np.min(grid_samples[i, j])
            delta_chi2_c = c_alpha - chi2_i_min
            delta_chi2_i = chi2_i_min - chi2_g_min

            c[i, j] = delta_chi2_c
            acceptance[i, j] = delta_chi2_i - delta_chi2_c
    
    # Plot the acceptance region
    plt.imshow(acceptance)
    plt.colorbar()
    plt.xlabel("Mass")
    plt.ylabel("Sin^2(2θ)")
    plt.title(f"Acceptance Region for {alpha}")
    plt.show()

if __name__ == "__main__":
    main()