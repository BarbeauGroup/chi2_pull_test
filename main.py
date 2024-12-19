from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

from transform_functions import csi
from transform_functions import pb_glass

from minimization.minimization import marginalize_mass_uu
from minimization.feldmancousins import feldmancousins, evaluate_gridpoint
from plotting.observables import plot_histograms
from transform_functions import form_factors

import numpy as np

from scipy.optimize import Bounds
import iminuit


def main():
    # ensemble = Ensemble("config/ensemble.json")

    # csi_oneton = Experiment("config/csi_asimov.json", csi.energy_efficiency, csi.time_efficiency, True, False, True)
    # pbglass20 = Experiment("config/pb_glass_20m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False, True)
    # pbglass30 = Experiment("config/pb_glass_30m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False, True)
    # pbglass40 = Experiment("config/pb_glass_40m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False, True)
    csi_real = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True, True, False, form_factors.helm)

    # ensemble.add_experiment(pbglass20)
    # ensemble.add_experiment(pbglass30)
    # ensemble.add_experiment(pbglass40)
    # ensemble.set_nuisance_params(["flux", "flux_pbxs"])
    # marginalize_mass_uu(ensemble, [0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/pb_glass3_2dssb")

    # print(np.logspace(-3, 0, num=100, endpoint=True))
    # ensemble2 = Ensemble("config/ensemble.json")
    # ensemble2.add_experiment(csi_oneton)
    # ensemble2.set_nuisance_params(["flux"])
    # # marginalize_mass_uu(ensemble2, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/csi_1t")
    # bounds = Bounds([1.370383450663168, 0.14174741629268048, 0.84, -np.inf], [1.370383450663168, 0.14174741629268048, 0.84, np.inf])
    # res = iminuit.minimize(ensemble2, [1.370383450663168, 0.14174741629268048, 0.84, 0], bounds=bounds)
    # print(res)

    # ensemble3 = Ensemble("config/ensemble.json")
    # ensemble3.set_nuisance_params(["flux", "flux_pbxs"])
    # ensemble3.add_experiment(csi_oneton)
    # ensemble3.add_experiment(pbglass20)
    # ensemble3.add_experiment(pbglass30)
    # ensemble3.add_experiment(pbglass40)
    # # marginalize_mass_uu(ensemble3, [0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/combined")

    ensemble = Ensemble("config/ensemble.json")
    ensemble.add_experiment(csi_real)
    ensemble.set_nuisance_params(["flux", 
                                  "flux_qf_csi",
                                  "nu_time_offset", 
                                  "brn_time_offset_csi",
                                  "nin_time_offset_csi",
                                  "brn_csi", 
                                  "nin_csi", 
                                  "ssb_csi",
                                  "r_n_csi"])
    # ensemble.set_nuisance_params([])
    # bounds = Bounds([0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], 
    #                 [100, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    # res = iminuit.minimize(ensemble, [0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0], bounds=bounds)
    # print(res)
    x0 = [0, 0, 0, 100, 100, 100, 0, 0, 0, 0]
    print(evaluate_gridpoint(5, 5, cost=ensemble, x0=x0, sin_bins=np.logspace(-3, 0, num=10, endpoint=True), mass_bins=np.logspace(0, 2, num=10), angle="me"))
    # feldmancousins(ensemble, [0, 0, 0, 100, 100, 100, 0, 0, 0, 0], np.logspace(-3, 0, num=10, endpoint=True), np.logspace(0, 2, num=10), "me", "output/real_csi")
    return 
    # print(-2 * (ensemble4(res.x) - ensemble4([0, 0, 0, 0, 0, 0, 0, 0])))
    # marginalize_mass_uu(ensemble4, [0, 80, 0, 0, 0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=20), "output/real_csi")

    ensemble4 = Ensemble("config/ensemble.json")
    parameters = {
        "mass": 0, #1.370,
        "ue4": 0, #0.1417,
        "umu4": 0, #0.8583,
        "nu_time_offset": 0,
        "brn_time_offset_csi": 0,
        "nin_time_offset_csi": 0,
        "flux": 0, #0.10917781,
        "brn_csi": 0.0,
        "nin_csi": 0.0,
        "ssb_csi": 0.0,
        "r_n_csi": 0.0
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
    # ensemble4.add_experiment(pbglass20)
    # hist_osc_1d, hist_unosc_1d = ensemble4.histograms(pbglass20, parameters)
    # plot_histograms(pbglass20.params, hist_unosc_1d, hist_osc_1d, alpha)

    ensemble4.add_experiment(csi_real)
    hist_osc_1d, hist_unosc_1d = ensemble4.histograms(csi_real, parameters)
    plot_histograms(csi_real.params, hist_unosc_1d, hist_osc_1d, alpha)

    e = 0
    t = 0
    for k in hist_unosc_1d["neutrinos"].keys():
        e += np.sum(hist_unosc_1d["neutrinos"][k]["energy"])
        t += np.sum(hist_unosc_1d["neutrinos"][k]["time"])
    print(e, t)

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