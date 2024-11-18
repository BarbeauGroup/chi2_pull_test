import numpy as np
from scipy.special import gammaln

def loglike_stat(histograms: dict, nuisance_params: dict) -> float:
    loglike = 0

    # Coincident Data
    for hist in ["energy", "time"]:
        ssb = histograms["ssb"][hist]
        observed = histograms["beam_state"]["C"][hist]

        predicted = 0
        for flavor in histograms["neutrinos"].keys():
            if flavor == "nuS" or flavor == "nuSBar":
                continue
            predicted += histograms["neutrinos"][flavor][hist] * (1 + nuisance_params[0])
        predicted += histograms["backgrounds"]["brn"][hist] * (1 + nuisance_params[1])
        predicted += histograms["backgrounds"]["nin"][hist] * (1 + nuisance_params[2])
        predicted += ssb * (1 + nuisance_params[3])

        loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))
    
    # Anti-Coincident Data
    for hist in ["energy", "time"]:
        ssb = histograms["ssb"][hist]
        observed = histograms["beam_state"]["AC"][hist]

        predicted = ssb * (1 + nuisance_params[3])

        loglike += np.sum(-predicted + observed*np.log(predicted) - gammaln(observed + 1))

    return loglike

def loglike_sys(nuisance_params: dict, nuisance_param_priors: dict) -> float:
    loglike = 0
    for i in range(len(nuisance_params)):
        loglike += -0.5*(nuisance_params[i]/nuisance_param_priors[i])**2 - np.log(np.sqrt(2*np.pi*nuisance_param_priors[i]**2))
    return loglike