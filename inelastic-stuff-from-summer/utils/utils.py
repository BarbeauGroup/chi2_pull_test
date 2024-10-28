import numpy as np


def P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14):
    return 1 - sin_squared_2_theta_14 * np.sin( 1.27*delta_m14_squared * L / E)**2

def toy_model(E, L, delta_m14_squared, sin_squared_2_theta_14, SM_prediction):
    return SM_prediction * P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14)

def flux_dependent_prediction(E, L, delta_m14_squared, sin_squared_2_theta_14, flux_norm_param):
    return flux_norm_param * P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14)


def chi2_stat(experiment, toy_model, prediction_parameters, sytematic_error_arr):
    
    # get the set observed value
    observed = experiment.get_n_observed()
    
    # set steady state background to a ratio of observed for now
    background = experiment.get_steady_state_background()*observed

    predicted = toy_model(*prediction_parameters)

    num = (observed - predicted*(1 + np.sum(sytematic_error_arr)))**2
    denom = np.sqrt(observed + 2*background)

    return num/denom


def chi2_sys(systematic_error_arr, systematic_error_dict):

    chi2 = 0

    i = 0
    for key in systematic_error_dict:
        
        nuisance_param_value = systematic_error_arr[i]
        nuisance_param_error = systematic_error_dict[key]

        chi2 += (nuisance_param_value/nuisance_param_error)**2

        i += 1
    
    return chi2


def chi2_total(experiment, toy_model, prediction_parameters, systematic_error_arr, systematic_error_dict):

    return chi2_stat(experiment, toy_model, prediction_parameters, systematic_error_arr) + chi2_sys(systematic_error_arr, systematic_error_dict)
