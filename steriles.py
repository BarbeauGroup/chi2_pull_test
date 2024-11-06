
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
    flux = read_flux_from_root(params["flux_file"])

    # create_observables(flux=flux, params=params)

    # Sterile Osc Parameters (temp)
    L = 19.3
    deltam41 = 1
    Ue4 = 0.3162
    Umu4 = 0.1778/3
    Utau4 = 0.0

    oscillated_flux = oscillate_flux(flux=flux)
    print(oscillated_flux)
    return
    
if __name__ == "__main__":
    main()