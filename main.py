from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

from transform_functions import csi
from transform_functions import pb_glass

from minimization.minimization import marginalize_mass_u, marginalize_mass_uu

import numpy as np


def main():
    ensemble = Ensemble("config/ensemble.json")

    # CsI = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True)

    csi_oneton = Experiment("config/csi_asimov.json", csi.energy_efficiency, csi.time_efficiency, True, False, True)
    pbglass20 = Experiment("config/pb_glass_20m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False)
    pbglass30 = Experiment("config/pb_glass_30m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False)
    pbglass40 = Experiment("config/pb_glass_40m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False, False)

    ensemble.add_experiment(pbglass20)
    ensemble.add_experiment(pbglass30)
    ensemble.add_experiment(pbglass40)
    ensemble.add_experiment(csi_oneton)

    marginalize_mass_uu(ensemble, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=30), "output/combined")

    ensemble2 = Ensemble("config/ensemble.json")
    ensemble2.add_experiment(csi_oneton)
    marginalize_mass_uu(ensemble2, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=30), "output/csi_1t")

    ensemble3 = Ensemble("config/ensemble.json")
    ensemble3.add_experiment(pbglass20)
    ensemble3.add_experiment(pbglass30)
    ensemble3.add_experiment(pbglass40)
    marginalize_mass_uu(ensemble3, [0], np.logspace(-3, 0, num=100, endpoint=True), np.logspace(-1, 1.7, num=30), "output/pb_glass3")

if __name__ == "__main__":

    main()