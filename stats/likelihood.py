import numpy as np
from scipy.special import gammaln

def loglike_stat_asimov(histograms_osc: dict, histograms_unosc: dict, nuisance_params: dict, factor=10) -> float:
    loglike = 0

    observed = 0
    for flavor in histograms_unosc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        observed += histograms_unosc["neutrinos"][flavor]
    observed += histograms_unosc["backgrounds"]["brn"]
    observed += histograms_unosc["backgrounds"]["nin"]
    observed += histograms_unosc["ssb"]
    observed *= factor

    predicted = 0
    for flavor in histograms_osc["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        predicted += histograms_osc["neutrinos"][flavor] * (1 + nuisance_params[0])
    predicted += histograms_osc["backgrounds"]["brn"] * (1 + nuisance_params[1])
    predicted += histograms_osc["backgrounds"]["nin"] * (1 + nuisance_params[2])
    predicted += histograms_osc["ssb"] * (1 + nuisance_params[3])
    predicted *= factor

    loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))

    return loglike 


def loglike_stat(histograms: dict, nuisance_params: dict) -> float:
    loglike = 0

    # Coincident Data
    ssb = histograms["ssb"]
    observed = histograms["beam_state"]["C"]

    predicted = 0
    for flavor in histograms["neutrinos"].keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        predicted += histograms["neutrinos"][flavor] * (1 + nuisance_params[0])
    predicted += histograms["backgrounds"]["brn"] * (1 + nuisance_params[1])
    predicted += histograms["backgrounds"]["nin"] * (1 + nuisance_params[2])
    predicted += ssb * (1 + nuisance_params[3])

    loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))
    
    # Anti-Coincident Data
    # ssb = histograms["ssb"]
    # observed = histograms["beam_state"]["AC"]

    # predicted = ssb * (1 + nuisance_params[3])

    # loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))

    return loglike

def loglike_sys(nuisance_params: dict, nuisance_param_priors: dict) -> float:
    loglike = 0
    for i in range(len(nuisance_params)):
        loglike += -0.5*(nuisance_params[i]/nuisance_param_priors[i])**2 - np.log(np.sqrt(2*np.pi*nuisance_param_priors[i]**2))
    return loglike