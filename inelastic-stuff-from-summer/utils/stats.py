import numpy as np


def chi2_stat(experiment, model, model_parameters, systematic_error_arr):
    
    # get the set observed value
    observed = experiment.get_n_observed()
    
    # set steady state background to a ratio of observed for now
    background = experiment.get_steady_state_background()*observed
    predicted = model(*model_parameters)

    num = (observed - predicted*(1 + np.sum(systematic_error_arr)))**2
    denom = np.power(np.sqrt(observed + 1*background),2)

    return num/denom


def chi2_stat_ratio(experiment1, experiment2, model, model_parameters):

    # get the set observed value
    s1 = experiment1.get_n_observed()
    s2 = experiment2.get_n_observed()

    # set steady state background to a ratio of observed for now
    b1 = experiment1.get_steady_state_background()
    b2 = experiment2.get_steady_state_background()

    p1 = model(experiment1, *model_parameters)
    p2 = model(experiment2, *model_parameters)

    # ratio
    ratio_observed = s1/s2
    ratio_predicted = p1/p2

    sigma_stat = np.sqrt( (s1 + s2)/(s2 + b2)**2 + (s1 + b1)*(s2 + b2)/(s2 + b2)**4)

    print("Ratio observed: ", ratio_observed)
    print("Ratio predicted: ", ratio_predicted)
    print("Sigma stat: ", sigma_stat)

    num = (ratio_observed - ratio_predicted)**2
    denom = sigma_stat**2

    return num/denom


def chi2_sys(experiment, systematic_error_arr):


    systematic_error_dict = experiment.get_systematic_error_dict()

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
