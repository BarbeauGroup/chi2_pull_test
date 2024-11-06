
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import pandas as pd

from numpy import heaviside

import numpy as np

import uproot

from flux.probabilities import Pab, sin2theta
from flux.nuflux import read_flux_from_root, oscillate_flux
from utils.loadparams import load_params
from flux.create_observables import create_observables

def main():
    params = load_params("config/csi.json")
    flux = read_flux_from_root(params)
    oscillated_flux = oscillate_flux(flux=flux)

    un_osc_obs = create_observables(params=params, flux=flux)
    osc_obs = create_observables(params=params, flux=oscillated_flux)


    
if __name__ == "__main__":
    main()