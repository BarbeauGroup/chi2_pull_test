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

    # ensemble.add_experiment(pbglass20)
    # ensemble.add_experiment(pbglass30)
    ensemble.add_experiment(csi_oneton)

    print(ensemble([0,0], 1, 0.01, 0.01))


    # marginalize_mass_u(ensemble, [0, 0, 0], np.linspace(0.01, 1, 1000), np.linspace(0, 50, 100), "output/pbglass")
    marginalize_mass_uu(ensemble, [0, 0], np.logspace(-2, 0, num=10), np.linspace(1, 1, 1), "output/csi_1t")

if __name__ == "__main__":
    main()