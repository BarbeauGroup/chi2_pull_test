import numpy as np

def sin2theta(alpha, beta, Ue4, Umu4, Utau4):
    """
    Calculate sin^2(theta) from mixing angles and mixing matrix elements.

    Parameters
    ----------
    alpha : int
        +1 for e, +2 for mu, +3 for tau, +4 for sterile
    beta : int
        +1 for e, +2 for mu, +3 for tau, +4 for sterile
    Ue4 : complex
        Mixing matrix element Ue4 squared.
    Umu4 : complex
        Mixing matrix element Umu4 squared.
    Utau4 : complex
        Mixing matrix element Utau4 squared.

    Returns
    -------
    float
        The sin^2(2theta(ab)).
    """
    if (alpha != 1 and alpha != 2 and alpha != 3 and alpha != 4) or (beta != 1 and beta != 2 and beta != 3 and beta != 4):
        raise ValueError("alpha and beta must be 1, 2, 3 or 4.")
    Us4 = 1 - Ue4 - Umu4 - Utau4
    delta = (1 if alpha == beta else 0)
    Ua4 = (Ue4 if alpha == 1 else Umu4 if alpha == 2 else Utau4 if alpha == 3 else Us4)
    Ub4 = (Ue4 if beta == 1 else Umu4 if beta == 2 else Utau4 if beta == 3 else Us4)
    return 4 * np.abs(
        delta*Ua4
        - Ua4*Ub4)

def Pab(Enu, L, deltam41, alpha, beta, Ue4, Umu4, Utau4):
    """
    Calculate the oscillation probability P_ab.
    
    Parameters
    ----------
    Enu : float
        Neutrino energy in MeV.
    L : float
        Baseline in m.
    deltam41 : float
        Mass squared difference in eV^2.
    alpha : int
        +1 for e, +2 for mu, +3 for tau, +4 for sterile
    beta : int
        +1 for e, +2 for mu, +3 for tau, +4 for sterile
    Ue4 : complex
        Mixing matrix element Ue4.
    Umu4 : complex
        Mixing matrix element Umu4.
    Utau4 : complex
        Mixing matrix element Utau4.

    Returns
    -------
    float
        The oscillation probability P_ab.
    """
    delta = (1 if alpha == beta else 0)
    return np.abs(delta
                  - sin2theta(alpha, beta, Ue4, Umu4, Utau4)
                    * np.sin(1.267 * deltam41**2 * L / Enu)**2)