import numpy as np
from scipy.special import gamma
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

import cProfile

def Pee(Enu, L, Ue4_2, deltam_2):
    return 1 - 4*(Ue4_2 - Ue4_2**2)*np.sin(1.267*deltam_2*L/Enu)**2

def EeeFlux(Enu, L, Ue4_2, deltam_2):
    mMu = 105.6583755  # MeV

    return (192./mMu) * (Enu/mMu)**2 *(0.5 - Enu/mMu) * Pee(Enu, L, Ue4_2, deltam_2)

def TtoQ(T, M):
    hbarc_fmMeV = 197.327  # fm MeV

    return np.sqrt(2*M*T + T**2) / hbarc_fmMeV

def rnToR0(rn):
    s = 0.3  # fm
    return np.sqrt(5./3. * (rn**2 - 3*s**2))

def j1(x):
    return (-x*np.cos(x) + np.sin(x))/(x**2)

def helmff(Q, rn):
    s = 0.3  # fm
    if Q == 0:
        return 1
    return (3*j1(Q*rnToR0(rn))/(Q*rnToR0(rn)))*np.exp(-Q**2 * s**2 / 2)

def F(T, rn, M):
    return helmff(TtoQ(T, M), rn)

def XSection(Enu, Er):
    # Constants and variables
    Gv = 0.0298 * 55 - 0.5117 * 78
    Ga = 0
    M = 123800.645  # MeV
    GF = 1.16637e-11  # MeV^-2
    rn = 4.77305  # fm
    a = 0.0749 / 1000
    b = 9.56 * 1000
    hbarc_cmMeV2 = 3.894e-22  # cm^2 MeV^2

    return (hbarc_cmMeV2 * GF**2 * M / (2*np.pi)) * (F(Er, rn, M)**2) * ( (Gv + Ga)**2 + (Gv - Ga)**2 * (1 - Er/Enu)**2 - (Gv**2 - Ga**2)*M*(Er / Enu**2))



def smear(x, e):
    a = 0.0749 / 1000
    b = 9.56 * 1000

    result = (
        (a / e * (1 + b * e))**(1 + b * e) / gamma(1 + b * e)
        * x**(b * e)
        * np.exp(-a / e * (1 + b * e) * x)
    )

    return result

def quenching_factor(Erec):
    """
    Calculate the quenching factor.

    Parameters
    ----------
    Erec : float
        Recoil energy in MeV (Enr)

    Returns
    -------
    float
        The observed energy in MeV (Eee)
    """

    return 0.0554628*Erec + 4.30681*np.power(Erec, 2) - 111.707 * np.power(Erec, 3) + 840.384 * np.power(Erec, 4)

def main():
    M = 123800.645  # MeV
    mMu = 105.6583755  # MeV

    max_nuE = mMu/2.
    max_recoil_nr = 2*(max_nuE)**2/M

    result = dblquad(lambda Enu,Er: EeeFlux(Enu, 19.3, 0, 0)*XSection(Enu, Er)*smear(10, quenching_factor(Er)), 0, max_recoil_nr, lambda Er: np.sqrt(M*Er/2), lambda Er: max_nuE)
    print("result: ", result)


    return
    result = quad(lambda er: er_spectrum(er) * smear(10, quenching_factor(er)), 0, max_recoil_nr)
    print(result)

    pe_spectrum = lambda pe: quad(lambda er: er_spectrum(er) * smear(pe, quenching_factor(er)), 0, max_recoil_nr)[0]

    # letsa plot

    pe = np.linspace(0, 50, 100)
    pe_spec = [pe_spectrum(p) for p in pe]

    # plt.plot(pe, pe_spec)
    # plt.show()


    # print(er_spectrum(0.002))
    # print(smear(10, 0.008))

if __name__ == "__main__":
    cProfile.run("main()", "output.prof")