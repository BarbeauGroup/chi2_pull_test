from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

from transform_functions import csi
from transform_functions import pb_glass

from minimization.minimization import marginalize_mass_uu
from plotting.observables import plot_histograms

import numpy as np


def main():
    # ensemble = Ensemble("config/ensemble.json")

    csi_oneton = Experiment("config/csi_asimov.json", csi.energy_efficiency, csi.time_efficiency, True, False, True)
    pbglass20 = Experiment("config/pb_glass_20m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False, True)
    pbglass30 = Experiment("config/pb_glass_30m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False, True)
    pbglass40 = Experiment("config/pb_glass_40m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False, True)
    # csi_real = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True, True)

    # ensemble.add_experiment(pbglass20)
    # ensemble.add_experiment(pbglass30)
    # ensemble.add_experiment(pbglass40)
    # ensemble.set_nuisance_params(["flux", "flux_pbxs"])
    # marginalize_mass_uu(ensemble, [0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/pb_glass3_2dssb")

    # ensemble2 = Ensemble("config/ensemble.json")
    # ensemble2.add_experiment(csi_oneton)
    # ensemble2.set_nuisance_params(["flux"])
    # marginalize_mass_uu(ensemble2, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/csi_1t")

    ensemble3 = Ensemble("config/ensemble.json")
    ensemble3.set_nuisance_params(["flux", "flux_pbxs"])
    ensemble3.add_experiment(csi_oneton)
    ensemble3.add_experiment(pbglass20)
    ensemble3.add_experiment(pbglass30)
    ensemble3.add_experiment(pbglass40)
    marginalize_mass_uu(ensemble3, [0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/combined")

    # ensemble4 = Ensemble("config/ensemble.json")
    # ensemble4.add_experiment(csi_real)
    # ensemble4.set_nuisance_params(["flux", "nu_time_offset", "brn_csi", "nin_csi", "ssb_csi"])
    # bounds = Bounds([0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [100, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf])
    # res = iminuit.minimize(ensemble4, [0, 0, 0, 0, 100, 0, 0, 0], bounds=bounds)
    # print(res)
    # print(-2 * (ensemble4(res.x) - ensemble4([0, 0, 0, 0, 0, 0, 0, 0])))
    # marginalize_mass_uu(ensemble4, [0, 80, 0, 0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/real_csi")

    ensemble4 = Ensemble("config/ensemble.json")
    parameters = {
        "mass": 3.0,
        "ue4": 0.5,
        "umu4": 0.5,
        "nu_time_offset": 0,
        "flux": 0.0,
        "brn_csi": 0.0,
        "nin_csi": 0.0,
        "ssb_csi": 0.0
    }

    # print("thetaee", sin2theta(1, 1, parameters["ue4"], parameters["umu4"], 0))
    # print("thetamue", sin2theta(1, 2, parameters["ue4"], parameters["umu4"], 0))
    # print("thetamumu", sin2theta(2, 2, parameters["ue4"], parameters["umu4"], 0))

    alpha = [
        parameters["flux"],
        parameters["brn_csi"],
        parameters["nin_csi"],
        parameters["ssb_csi"]
    ]
    ensemble4.add_experiment(pbglass20)
    hist_osc_1d, hist_unosc_1d = ensemble4.histograms(pbglass20, parameters)
    plot_histograms(pbglass20.params, hist_unosc_1d, hist_osc_1d, alpha)

    ensemble4.add_experiment(csi_oneton)
    hist_osc_1d, hist_unosc_1d = ensemble4.histograms(csi_oneton, parameters)
    plot_histograms(csi_oneton.params, hist_unosc_1d, hist_osc_1d, alpha)

    # ensemble5 = Ensemble("config/ensemble.json")
    # parameters = {
    #     "mass": 0.0,
    #     "ue4": 0.0,
    #     "umu4": 0.0,
    #     "nu_time_offset": 0,
    #     "flux": 0.0,
    # }
    # ensemble5.add_experiment(pbglass40)
    # ensemble5.set_nuisance_params(["flux", "flux_pbxs"])
    # print(ensemble5([0, 0], 0, 0, 0))
    # hist_osc_1d, hist_unosc_1d = ensemble5.histograms(pbglass40, parameters) # Should probably apply the nuisance parameters here


    # print(hist_osc_1d)

if __name__ == "__main__":
    main()