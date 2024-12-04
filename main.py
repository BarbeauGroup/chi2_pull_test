from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

from transform_functions import csi
from transform_functions import pb_glass

from minimization.minimization import marginalize_mass_uu
from plotting.observables import plot_histograms

from utils.data_loaders import read_flux_from_root

from transform_functions import csi
from transform_functions import pb_glass

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins, project_histograms, plot_csi

import numpy as np


def main():
    ensemble = Ensemble("config/ensemble.json")

    # CsI = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True)

    csi_oneton = Experiment("config/csi_asimov.json", csi.energy_efficiency, csi.time_efficiency, True, True, True)
    pbglass20 = Experiment("config/pb_glass_20m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False)
    pbglass30 = Experiment("config/pb_glass_30m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False)
    pbglass40 = Experiment("config/pb_glass_40m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False)
    csi_real = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True, True)

    ensemble.add_experiment(pbglass20)
    ensemble.add_experiment(pbglass30)
    ensemble.add_experiment(pbglass40)
    # ensemble.add_experiment(csi_oneton)
    ensemble.set_nuisance_params(["flux", "flux_pbxs"])

    # marginalize_mass_uu(ensemble, [0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=30), "output/pb_glass3")

    # ensemble2 = Ensemble("config/ensemble.json")
    # ensemble2.add_experiment(csi_oneton)
    # ensemble2.set_nuisance_params(["flux"])
    # marginalize_mass_uu(ensemble2, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=30), "output/csi_1t")

    # ensemble3 = Ensemble("config/ensemble.json")
    # ensemble3.set_nuisance_params(["flux"])
    # ensemble3.add_experiment(pbglass20)
    # ensemble3.add_experiment(pbglass30)
    # ensemble3.add_experiment(pbglass40)
    # marginalize_mass_uu(ensemble3, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=30), "output/pb_glass3")

    ensemble4 = Ensemble("config/ensemble.json")
    parameters = {
        "mass": 3.0,
        "ue4": 0.5,
        "umu4": 0.0,
        "flux_time_offset": 80,
        "flux": 0.0,
        "brn_csi": 0.0,
        "nin_csi": 0.0,
        "ssb_csi": 0.0
    }
    alpha = [
        parameters["flux"],
        parameters["brn_csi"],
        parameters["nin_csi"],
        parameters["ssb_csi"]
    ]
    ensemble4.add_experiment(pbglass20)
    hist_osc_1d, hist_unosc_1d = ensemble4.histograms(pbglass20, parameters)
    plot_histograms(pbglass20.params, hist_unosc_1d, hist_osc_1d, alpha)

    # ensemble5 = Ensemble("config/ensemble.json")
    # parameters = {
    #     "mass": 0.0,
    #     "ue4": 0.0,
    #     "umu4": 0.0,
    #     "flux_time_offset": 0,
    #     "flux": 0.0,
    # }
    # ensemble5.add_experiment(pbglass40)
    # hist_osc_1d, hist_unosc_1d = ensemble5.histograms(pbglass40, parameters) # Should probably apply the nuisance parameters here


    # print(hist_osc_1d)

if __name__ == "__main__":

    main()