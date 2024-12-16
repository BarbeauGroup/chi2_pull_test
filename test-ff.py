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

def test():
    i127 = np.load("data/flux_transfer_matrices/csi_I127_flux_smearing_matrix_unity.npy")
    recoil_bins = np.linspace(0, 5885 * 0.01, 5885)
    isotope = {"A": 127, "mass": 126.9}
    ff_2 = form_factors.helm(isotope, recoil_bins, 0)**2
    ffed = i127 * ff_2[:, None]
    np.savetxt("test.csv", ffed[:, 58])
    
if __name__ == "__main__":
    # main()
    test()