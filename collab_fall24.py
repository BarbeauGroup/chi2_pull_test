from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

import scienceplots
import matplotlib.pyplot as plt

from transform_functions import csi

def main():


    ensemble = Ensemble("config/ensemble.json")
    ccsi = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True, True)
    ensemble.add_experiment(csi)

    unosc_flux = ensemble.oscillated_flux(ccsi, {"mass": 0, "ue4": 0, "umu4": 0.0})
    osc_flux = ensemble.oscillated_flux(ccsi, {"mass": 3.0, "ue4": 0.3, "umu4": 0.0})


    for key in unosc_flux.keys():
        bins = unosc_flux[key][0]
        t_bins = bins[0]
        e_bins = bins[1]

        counts = unosc_flux[key][1]

        t_counts = np.sum(counts, axis=0)
        e_counts = np.sum(counts, axis=1)


if __name__ == "__main__":
    main()