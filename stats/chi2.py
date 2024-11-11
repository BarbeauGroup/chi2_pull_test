import numpy as np


def chi2_stat(histograms: dict, nuisance_params: dict) -> float:
    """

    histograms: dictionary with keys "beam_state", "neutrinos", "backgrounds"

    returns: chi2 value

    """
    chi2 = 0
    for hist in ["energy", "time"]:

        ssb = histograms["beam_state"]["AC"][hist]
        observed = histograms["beam_state"]["C"][hist]

        predicted = (histograms["neutrinos"]["nuE"][hist] + histograms["neutrinos"]["nuMu"][hist] + histograms["neutrinos"]["nuTau"][hist]) * (1 + nuisance_params["flux"][0])
        predicted += histograms["backgrounds"]["brn"][hist] * (1 + nuisance_params["brn"][0])
        predicted += histograms["backgrounds"]["nin"][hist] * (1 + nuisance_params["nin"][0])
        predicted += ssb * (1 + nuisance_params["ssb"][0])


        num = (observed - predicted)**2
        denom = np.abs((observed - ssb) + 2*ssb + histograms["backgrounds"]["brn"][hist] + histograms["backgrounds"]["nin"][hist])

        chi2 += np.sum(num/denom)

        for param in nuisance_params.keys():
            chi2 += np.square(nuisance_params[param][0]/nuisance_params[param][1])

    return chi2



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


# def chi2_total(experiment, toy_model, prediction_parameters, systematic_error_arr, systematic_error_dict):

#     return chi2_stat(experiment, toy_model, prediction_parameters, systematic_error_arr) + chi2_sys(systematic_error_arr, systematic_error_dict)
