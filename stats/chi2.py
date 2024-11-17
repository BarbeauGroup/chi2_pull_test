import numpy as np


def chi2_stat(histograms: dict, nuisance_params: dict) -> float:
    """

    histograms: dictionary with keys "beam_state", "neutrinos", "backgrounds"

    returns: chi2 value

    """
    chi2 = 0

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

        num = (observed - predicted)**2
        denom = np.abs((observed - ssb) + 2*ssb + histograms["backgrounds"]["brn"][hist] + histograms["backgrounds"]["nin"][hist])

        chi2 += np.sum(num/denom)

    # Anti-Coincident Data
    for hist in ["energy", "time"]:

        ssb = histograms["ssb"][hist]
        observed = histograms["beam_state"]["AC"][hist]

        predicted = ssb * (1 + nuisance_params[3])

        num = (observed - predicted)**2
        denom = np.abs((observed - ssb) + 2*ssb + histograms["backgrounds"]["brn"][hist] + histograms["backgrounds"]["nin"][hist])

        chi2 += np.sum(num/denom)

    return chi2



def chi2_sys(nuisance_params: dict, nuisance_param_priors: dict) -> float:
    chi2 = 0
    for i in range(len(nuisance_params)):
        chi2 += (nuisance_params[i]/nuisance_param_priors[i])**2
    return chi2