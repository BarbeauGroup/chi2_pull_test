from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

from transform_functions import csi
from transform_functions import pb_glass

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins

from stats.likelihood import loglike_stat, loglike_sys

from minimization.minimization import marginalize_mass_u, marginalize_mass_uu

import numpy as np
from copy import deepcopy
from functools import partial


def main():
    ensemble = Ensemble("config/ensemble.json")

    # CsI = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True)
    pbglass20 = Experiment("config/pb_glass_20m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False)
    pbglass30 = Experiment("config/pb_glass_30m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, True, False)
    ensemble.add_experiment(pbglass20)
    ensemble.add_experiment(pbglass30)

    # marginalize_mass_u(ensemble, [0, 0, 0], np.linspace(0.01, 1, 1000), np.linspace(0, 50, 100), "output/pbglass")
    marginalize_mass_uu(ensemble, [0, 0], np.linspace(0.01, 1, 200), np.linspace(0, 5, 10), "output/pbglass")

if __name__ == "__main__":
    main()