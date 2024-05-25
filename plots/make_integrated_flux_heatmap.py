import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

sin_squared_2_theta_14 = np.load("data/oscillation_sin.npy")
delta_m14_squared = np.load("data/oscillation_delta_m.npy")
osc_fraction = np.load("data/oscillation_flux_arr.npy")



plt.style.use("science")

plt.figure(figsize=(8, 6))

plt.imshow(osc_fraction, cmap="viridis", origin="lower", aspect="auto")
plt.colorbar(label="Fraction of Oscillated Flux")

plt.xticks(np.arange(len(sin_squared_2_theta_14)), np.round(sin_squared_2_theta_14,2))
plt.yticks(np.arange(len(delta_m14_squared)), np.round(delta_m14_squared,2))

# Loop over data dimensions and create text annotations.
for i in range(len(delta_m14_squared)):
    for j in range(len(sin_squared_2_theta_14)):
        text = plt.text(j, i, np.round(osc_fraction[i, j], 3),
                       ha="center", va="center", color="black")

plt.xlabel(r"$\sin^2(2\theta_{14})$", fontsize=18)
plt.ylabel(r"$\Delta m^2_{14}$", fontsize=18)

plt.show()