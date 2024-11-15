# import unittest
# import numpy as np

# import sys
# sys.path.append("../")

# from flux.nuflux import oscillate_flux
# from utils.loadparams import load_params
# from flux.create_observables import create_observables

# class TestCreateObservables(unittest.TestCase):
#     flux = {}
#     params = {}
#     x = []
#     osc_params = []
#     oscillated_flux = {}
#     un_osc_obs = {}
#     osc_obs = {}

#     def setUp(self):
#         with open("flux/flux_dict.pkl", "rb") as f:
#             self.flux = np.load(f, allow_pickle=True).item()
#         self.params = load_params("config/csi.json")

#         self.x = [1.521e+00, 4.174e-01, 4.462e-01, 1.196e-02, -4.793e-03,
#                  -4.465e-03, -1.447e-02]
    
#         self.osc_params = self.x[0:3]
#         self.osc_params = [self.params["detector"]["distance"]/100., self.osc_params[0], self.osc_params[1], self.osc_params[2], 0.0]

#         self.oscillated_flux = oscillate_flux(flux=self.flux, oscillation_params=self.osc_params)

#         self.osc_obs = create_observables(params=self.params, flux=self.oscillated_flux)
#         self.un_osc_obs = create_observables(params=self.params, flux=self.flux)
        
#         return super().setUp()

#     def test_create_observables(self):
#         energy_osc = 0
#         time_osc = 0
#         energy_unosc = 0
#         time_unosc = 0
#         osc = 0
#         unosc = 0
#         for flavor in self.osc_obs.keys():
#             osc += self.osc_obs[flavor]["energy"][1]
#             unosc += self.un_osc_obs[flavor]["energy"][1]
#             energy_osc += np.sum(self.osc_obs[flavor]["energy"][1])
#             energy_unosc += np.sum(self.un_osc_obs[flavor]["energy"][1])
#             time_osc += np.sum(self.osc_obs[flavor]["time"][1])
#             time_unosc += np.sum(self.un_osc_obs[flavor]["time"][1])
#         print(np.sum(osc), np.sum(unosc))
#         # print(energy_osc, time_osc)
#         self.assertTrue(np.isclose(energy_osc, time_osc, rtol=1e-2))
#         self.assertTrue(np.isclose(energy_unosc, time_unosc, rtol=1e-2))
#         print(osc, unosc)
#         self.assertTrue(np.allclose(osc, unosc, rtol=1e-3))


#         osc = 0
#         unosc = 0
#         for flavor in self.osc_obs.keys():
#             osc += self.osc_obs[flavor]["time"][1]
#             unosc += self.un_osc_obs[flavor]["time"][1]
#         print(osc, unosc)
#         self.assertTrue(np.allclose(osc, unosc))