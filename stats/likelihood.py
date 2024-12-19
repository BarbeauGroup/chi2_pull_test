import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2, norm

def dll_to_sigma(dll, dof=1):
    percent = chi2.cdf(dll, dof)
    return norm.ppf(percent)

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

    with np.errstate(all='raise'):
        try:
            loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))
        except FloatingPointError:
            # print(np.sum(predicted), np.sum(observed), nuisance_params)
            return -np.inf

    return loglike 


def loglike_stat(histograms: dict, nuisance_params: dict, exp_name: str) -> float:
    loglike = 0

    # Coincident Data
    observed = histograms["beam_state"]["C"]

    predicted = 0
    for flavor in histograms["neutrinos"].keys():
        if flavor != "combined":
            continue

        flux_nuisance = 1
        for k in nuisance_params.keys():
            if k.startswith("flux"):
                flux_nuisance += nuisance_params[k]

        predicted += histograms["neutrinos"][flavor] * flux_nuisance

    for key in histograms.keys():
        if key == "neutrinos" or key == "beam_state":
            continue
        predicted += histograms[key] * (1 + nuisance_params.get(f"{key}_{exp_name}", 0))

    with np.errstate(all='raise'):
        try:
            loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))
        except FloatingPointError:
            # print(np.sum(predicted), np.sum(observed), nuisance_params)
            return -np.inf
    
    # Anti-Coincident Data
    # ssb = histograms["ssb"]
    # observed = histograms["beam_state"]["AC"]

    # predicted = ssb * (1 + nuisance_params[3])

    # loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))

    return loglike

def loglike_sys(nuisance_params: dict, nuisance_param_priors: dict) -> float:
    loglike = 0
    for k in nuisance_param_priors.keys():
        if hasattr(nuisance_param_priors[k], "__len__"): continue # skip the nuisance parameters with uniform priors
        loglike += -0.5*(nuisance_params[k]/nuisance_param_priors[k])**2 - np.log(np.sqrt(2*np.pi*nuisance_param_priors[k]**2))
    return loglike