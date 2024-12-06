import json
import sys
import numpy as np

sys.path.append('../')
from utils.calcs import num_atoms
from utils.data_loaders import read_brns_nins_from_txt, read_data_from_txt
from flux.ssb_pdf import make_ssb_pdf

class Experiment:
    def __init__(self, config_file, e_eff, t_eff, asimov, bkgs_exist, ssb_exists, flat_ssb=False):
        with open(config_file, 'r') as f:
            self.params = json.load(f)
        
        self.flux_matrix = np.load(self.params["detector"]["flux_matrix"])
        self.detector_matrix = np.load(self.params["detector"]["detector_matrix"])
        self.matrix = self.detector_matrix @ self.flux_matrix

        self.num_atoms = num_atoms(self.params)

        self.observable_energy_bins = np.asarray(self.params["analysis"]["energy_bins"])
        self.observable_time_bins = np.asarray(self.params["analysis"]["time_bins"])

        self.asimov = asimov
        if not asimov: self.data_dict = read_data_from_txt(self.params)
        else: self.data_dict = None

        self.ssb_exists = ssb_exists
        if ssb_exists: self.ssb_dict = make_ssb_pdf(self.params)
        else: self.ssb_dict = None

        self.flat_ssb = flat_ssb
        if self.ssb_exists and self.flat_ssb:
            raise ValueError("Cannot have a flat SSB and a SSB PDF at the same time")
        
        self.bkgs_exist = bkgs_exist
        if bkgs_exist: self.bkd_dict = read_brns_nins_from_txt(self.params)
        else: self.bkd_dict = None
        if self.bkgs_exist and self.asimov:
            raise ValueError("Cannot have background data and an Asimov data set at the same time right now")

        self.energy_efficiency = e_eff
        self.time_efficiency = t_eff