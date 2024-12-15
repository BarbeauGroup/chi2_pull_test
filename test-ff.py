from classes.Experiment import Experiment
from classes.Ensemble import Ensemble

from transform_functions import csi
from transform_functions import pb_glass
from transform_functions import form_factors

from minimization.minimization import marginalize_mass_uu
from plotting.observables import plot_histograms

import numpy as np


def main():
    ensemble = Ensemble("config/ensemble.json")
    csi_real = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, False, True, True, False, form_factors.helm)
    ensemble.add_experiment(csi_real)

    ensemble.set_nuisance_params(["flux", "nu_time_offset", "brn_csi", "nin_csi", "ssb_csi"])

    ensemble([0, 0, 0, 0, 0, 0, 0, 0], 1.0, 1.0, 1.0)

if __name__ == "__main__":
    main()