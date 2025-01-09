import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt

import cProfile

def er_spectrum(x):
    # Constants and variables
    Gv = 0.0298 * 55 - 0.5117 * 78
    Ga = 0
    M = 123800.645  # MeV
    GF = 1.16637e-11  # MeV^-2
    mMu = 105.6583755  # MeV
    s = 0.3  # fm
    rn = 4.77305  # fm
    a = 0.0749 / 1000
    b = 9.56 * 1000
    hbarc_cmMeV2 = 3.894e-22  # cm^2 MeV^2
    hbarc_fmMeV = 197.327  # fm MeV

    result = (
        -(
            (27 * np.exp(-(s**2 * x * (2 * M + x) / hbarc_fmMeV**2))
            * GF**2 * hbarc_fmMeV**4 * hbarc_cmMeV2 * M
            * (
                -8 * Ga * Gv * x * (
                    mMu**2 * (mMu - 3 * x)
                    + 6 * np.sqrt(2) * mMu * x * np.sqrt(M * x)
                    + 4 * np.sqrt(2) * (M * x)**(3/2)
                    - 6 * M * x * (mMu + x)
                )
                - Gv**2 * (
                    mMu**4
                    - 4 * mMu**3 * x
                    + 12 * mMu**2 * x * (-M + x)
                    - 4 * M * x**2 * (3 * M - 6 * x + 4 * np.sqrt(2) * np.sqrt(M * x))
                    + 8 * mMu * x * (
                        3 * M * x
                        + 2 * np.sqrt(2) * M * np.sqrt(M * x)
                        - 3 * np.sqrt(2) * x * np.sqrt(M * x)
                    )
                )
                - Ga**2 * (
                    mMu**4
                    - 4 * mMu**3 * x
                    + 12 * mMu**2 * x * (M + x)
                    + 4 * M * x**2 * (9 * M + 6 * x - 4 * np.sqrt(2) * np.sqrt(M * x))
                    - 8 * mMu * x * (
                        -3 * M * x
                        + 4 * np.sqrt(2) * M * np.sqrt(M * x)
                        + 3 * np.sqrt(2) * x * np.sqrt(M * x)
                    )
                )
            )
            * (
                np.sqrt(15) * np.sqrt(rn**2 - 3 * s**2) * np.sqrt(x * (2 * M + x))
                * np.cos((np.sqrt(5/3) * np.sqrt(rn**2 - 3 * s**2) * np.sqrt(x * (2 * M + x))) / hbarc_fmMeV)
                - 3 * hbarc_fmMeV * np.sin((np.sqrt(5/3) * np.sqrt(rn**2 - 3 * s**2) * np.sqrt(x * (2 * M + x))) / hbarc_fmMeV)
            )**2
            )
        ) / (
            125 * mMu**4 * np.pi * (rn**2 - 3 * s**2)**3 * x**3 * (2 * M + x)**3
        )
    )

    return result

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
    max_recoil_nr = 2*(mMu/2)**2/M
    print(max_recoil_nr)
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