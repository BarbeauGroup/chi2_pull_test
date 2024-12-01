import json
import sys
import numpy as np

sys.path.append('../')
from utils.calcs import num_atoms
from utils.data_loaders import read_brns_nins_from_txt, read_data_from_txt
from flux.ssb_pdf import make_ssb_pdf

class Experiment:
    def __init__(self, config_file, e_eff, t_eff, data_exists):
        with open(config_file, 'r') as f:
            self.params = json.load(f)
        
        self.flux_matrix = np.load(self.params["detector"]["flux_matrix"])
        self.detector_matrix = np.load(self.params["detector"]["detector_matrix"])
        # self.matrix = self.detector_matrix @ self.flux_matrix
        self.matrix = self.flux_matrix

        self.num_atoms = num_atoms(self.params)
        print(self.num_atoms)

        self.observable_energy_bins = np.asarray(self.params["analysis"]["energy_bins"])
        self.observable_time_bins = np.asarray(self.params["analysis"]["time_bins"])

        if data_exists:
            # load in brn and nin histograms
            self.bkd_dict = read_brns_nins_from_txt(self.params)

            # load in data and ssb histograms
            self.data_dict = read_data_from_txt(self.params)
            self.ssb_dict = make_ssb_pdf(self.params)

        self.energy_efficiency = e_eff
        self.time_efficiency = t_eff