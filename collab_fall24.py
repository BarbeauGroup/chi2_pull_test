from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

import scienceplots
import matplotlib.pyplot as plt

from transform_functions import csi
import numpy as np

def main():


    ensemble = Ensemble("config/ensemble.json")
    ccsi = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True, True)
    ensemble.add_experiment(csi)

    unosc_flux = ensemble.oscillated_flux(ccsi, {"mass": 0, "ue4": 0, "umu4": 0.0})
    osc_flux = ensemble.oscillated_flux(ccsi, {"mass": 3.0, "ue4": 0.15, "umu4": 0.15})


    label = {"nuE": r"$\nu_e$", "nuEBar": r"$\bar{\nu_e}$", 
             "nuMu": r"$\nu_\mu$", "nuMuBar": r"$\bar{\nu_\mu}$", 
             "nuTau": r"$\nu_\tau$", "nuTauBar": r"$\bar{\nu_\tau}$", 
             "nuS": r"$\nu_s$", "nuSBar": r"$\bar{\nu_s}$"}


    plt.figure(figsize=(10,6))

    for i,key in enumerate(unosc_flux.keys()):
        if key!="nuEBar": continue
        if key == "nuS" or key == "nuSBar": continue
        if key == "nuTau" or key == "nuTauBar": continue
        unosc_bins = unosc_flux[key][0]
        t_bins = unosc_bins[0]
        e_bins = unosc_bins[1]

        osc_bins = osc_flux[key][0]

        unosc_counts = unosc_flux[key][1]
        osc_counts = osc_flux[key][1]

        unosc_t_counts = np.sum(unosc_counts, axis=1)
        unosc_e_counts = np.sum(unosc_counts, axis=0)

        osc_t_counts = np.sum(osc_counts, axis=1)
        osc_e_counts = np.sum(osc_counts, axis=0)

   
        plt.plot(t_bins[:-1], unosc_t_counts, label=fr"{label[key]} SM", linestyle='--', alpha=0.5)
        # plt.plot(t_bins[:-1], osc_t_counts, label=fr"{label[key]} disappearance only")
        plt.plot(t_bins[:-1], osc_t_counts, label=fr"{label[key]} full mixing")



    plt.legend(fontsize=20)

    plt.xlabel("Time (ns)", fontsize=20)
    plt.xlim(0, 6000)

    # plt.title("Gallium Anomaly Parameters 19.3m from SNS Target", fontsize=20)
    plt.title(r"$|U_{e4}|^2 = 0.15 \quad |U_{\mu4}|^2 = 0.15$", fontsize=20)

    plt.show()


if __name__ == "__main__":
    main()