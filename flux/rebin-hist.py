import uproot
import numpy as np

from utils.histograms import rebin_histogram

rf = uproot.open("snsFlux2D.root")

print(rf.keys())

convolved_energy_and_time_of_nu_mu = rf["convolved_energy_time_of_nu_mu;1"]
convolved_energy_and_time_of_nu_mu_bar = rf["convolved_energy_time_of_anti_nu_mu;1"]
convolved_energy_and_time_of_nu_e = rf["convolved_energy_time_of_nu_e;1"]
convolved_energy_and_time_of_nu_e_bar = rf["convolved_energy_time_of_anti_nu_e;1"]

# TODO: put in config file
new_t_edges = np.arange(0, 15010, 10)

# nu e
NuE = convolved_energy_and_time_of_nu_e.values()
anc_keNuE_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()
anc_keNuE_values = np.sum(NuE, axis=0)

anc_tNuE_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()
anc_tNuE_values = np.sum(NuE, axis=1)
anc_tNuE_values = rebin_histogram(anc_tNuE_values, anc_tNuE_edges, new_t_edges)
anc_tNuE_values = anc_tNuE_values / np.sum(anc_tNuE_values)

# nu e bar
NuEBar = convolved_energy_and_time_of_nu_e_bar.values()
anc_keNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()
anc_keNuEBar_values = np.sum(NuEBar, axis=0)

anc_tNuEBar_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()
anc_tNuEBar_values = np.sum(NuEBar, axis=1)
anc_tNuEBar_values = rebin_histogram(anc_tNuEBar_values, anc_tNuEBar_edges, new_t_edges)
anc_tNuEBar_values = anc_tNuEBar_values / np.sum(anc_tNuEBar_values)

# nu mu
NuMu = convolved_energy_and_time_of_nu_mu.values()
anc_keNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()
anc_keNuMu_values = np.sum(NuMu, axis=0)

anc_tNuMu_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()
anc_tNuMu_values = np.sum(NuMu, axis=1)
anc_tNuMu_values = rebin_histogram(anc_tNuMu_values, anc_tNuMu_edges, new_t_edges)
anc_tNuMu_values = anc_tNuMu_values / np.sum(anc_tNuMu_values)

# nu mu bar
NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()
anc_keNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()
anc_keNuMuBar_values = np.sum(NuMuBar, axis=0)

anc_tNuMuBar_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()
anc_tNuMuBar_values = np.sum(NuMuBar, axis=1)
anc_tNuMuBar_values = rebin_histogram(anc_tNuMuBar_values, anc_tNuMuBar_edges, new_t_edges)
anc_tNuMuBar_values = anc_tNuMuBar_values / np.sum(anc_tNuMuBar_values)