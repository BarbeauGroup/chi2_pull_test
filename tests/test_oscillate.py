import unittest
import numpy as np

import sys
sys.path.append("../")

from flux.nuflux import oscillate_flux
from utils.loadparams import load_params

class TestOscillate(unittest.TestCase):
    flux = {}
    params = {}
    x = []
    osc_params = []
    oscillated_flux = {}

    def setUp(self):
        with open("flux/flux_dict.pkl", "rb") as f:
            self.flux = np.load(f, allow_pickle=True).item()
        self.params = load_params("config/csi.json")

        self.x = [1.521e+00, 4.174e-01, 4.462e-01, 1.196e-02, -4.793e-03,
                 -4.465e-03, -1.447e-02]
    
        self.osc_params = self.x[0:3]
        self.osc_params = [self.params["detector"]["distance"]/100., self.osc_params[0], self.osc_params[1], self.osc_params[2], 0.0]

        self.oscillated_flux = oscillate_flux(flux=self.flux, oscillation_params=self.osc_params)
        
        return super().setUp()
    
    def test_oscillate(self):
        osc = 0
        unosc = 0
        for flavor in self.flux.keys():
            osc += self.oscillated_flux[flavor][1]
            unosc += self.flux[flavor][1]
        self.assertTrue(np.allclose(osc, unosc))
