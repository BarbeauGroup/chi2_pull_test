import numpy as np

from flux.probabilities import Pab

def oscillate_flux(flux: dict, oscillation_params: dict) -> dict:
    """
    flux: dictionary with SNS flux information

    returns: dictionary with oscillated SNS flux information
    """

    ###

    # Oscillation Parameters

    L = oscillation_params["L"]
    deltam41 = oscillation_params["deltam41"]
    Ue4 = oscillation_params["Ue4"]
    Umu4 = oscillation_params["Umu4"]
    Utau4 = oscillation_params["Utau4"]

    # Make an empty dictionary to store the oscillated flux information
    oscillated_flux = {}

    flavors = ["nuE", "nuMu", "nuTau", "nuS"]
    anti_flavors = ["nuEBar", "nuMuBar", "nuTauBar", "nuSBar"]

    # This for loop fills the dictionary
    for final_i, (final_flavor, final_antiflavor) in enumerate(zip(flavors, anti_flavors)):
        oscillated_energy_flux = 0
        anti_oscillated_energy_flux = 0

        oscillated_time_flux = 0
        anti_oscillated_time_flux = 0

        # This for loop does the sum over initial flavors to calculate the oscillated flux
        for initial_i, (initial_flavor, initial_antiflavor) in enumerate(zip(flavors, anti_flavors)):
            temp = Pab(flux[initial_flavor]["energy"][0], L, deltam41, final_i+1, initial_i+1, Ue4, Umu4, Utau4) * flux[initial_flavor]["energy"][1]
            oscillated_energy_flux += temp
            if np.sum(flux[initial_flavor]["energy"][1]) != 0:
                oscillated_time_flux += np.sum(temp) / np.sum(flux[initial_flavor]["energy"][1]) * flux[initial_flavor]["time"][1]
            else:
                oscillated_time_flux += 0

            antitemp = Pab(flux[initial_antiflavor]["energy"][0], L, deltam41, final_i+1,  initial_i+1, Ue4, Umu4, Utau4) * flux[initial_antiflavor]["energy"][1]
            anti_oscillated_energy_flux += antitemp
            if np.sum(flux[initial_antiflavor]["energy"][1]) != 0:
                anti_oscillated_time_flux += np.sum(antitemp) / np.sum(flux[initial_antiflavor]["energy"][1]) * flux[initial_antiflavor]["time"][1]
            else:
                anti_oscillated_time_flux += 0

        if np.sum(oscillated_time_flux) != 0:
            oscillated_time_flux /= np.sum(oscillated_time_flux)
        if np.sum(anti_oscillated_time_flux) != 0:
            anti_oscillated_time_flux /= np.sum(anti_oscillated_time_flux)

        oscillated_flux[final_flavor] = {
            "energy" : (flux[final_flavor]["energy"][0], oscillated_energy_flux),
            "time" : (flux[final_flavor]["time"][0], oscillated_time_flux)
        }
        oscillated_flux[final_antiflavor] = {
            "energy" : (flux[final_antiflavor]["energy"][0], anti_oscillated_energy_flux),
            "time" : (flux[final_antiflavor]["time"][0], anti_oscillated_time_flux)
        }

    return oscillated_flux