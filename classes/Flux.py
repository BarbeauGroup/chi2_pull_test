import sys
import json
import numpy as np

sys.path.append('../')
from utils.data_loaders import read_flux_from_root

class Flux:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.params = json.load(f)
        
        time_edges = np.arange(*self.params["time_edges"])

        self.flux = read_flux_from_root(self.params, time_edges)