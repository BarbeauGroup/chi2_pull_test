import numpy as np
from scipy.special import gamma
from scipy.integrate import dblquad, tplquad
from scipy import LowLevelCallable
from scipy.stats import rv_continuous, expon
import matplotlib.pyplot as plt
import os, ctypes

lib = ctypes.CDLL(os.path.abspath('cfuncs.so'))
lib.recoil_spectrum.restype = ctypes.c_double
lib.recoil_spectrum.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

lib.recoil_spectrum_int.restype = ctypes.c_double
lib.recoil_spectrum_int.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

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

def p(x, theta):
    M = 123800.645  # MeV
    mMu = 105.6583755  # MeV

    max_nuE = mMu/2.
    max_recoil_nr = 2*(max_nuE)**2/M

    _theta = (ctypes.c_double*3)(x, theta[0], theta[1])
    user_data = ctypes.cast(_theta, ctypes.c_void_p)

    func = LowLevelCallable(lib.recoil_spectrum, user_data)

    return dblquad(func, 0, max_recoil_nr, lambda Er: np.sqrt(M*Er/2), lambda Er: max_nuE)[0]

    # return dblquad(lambda Enu,Er: EeeFlux(Enu, 19.3, theta[0], theta[1])
    #                                             * XSection(Enu, Er)
    #                                             * smear(x, quenching_factor(Er)),
    #                                             0, max_recoil_nr, lambda Er: np.sqrt(M*Er/2), lambda Er: max_nuE)[0]

def q(x):
    return expon.pdf(x, scale=10.)

def sample(size, theta):
    pe = np.linspace(0, 20, 100)
    pe_spec = [p(p_, theta) for p_ in pe]
    max_p = np.max(pe_spec)

    out = []

    i = 0

    while len(out) < size:
        print(i)
        xs = np.random.exponential(10, size)
        cs = np.random.uniform(0, 1, size)
        k =  11. * max_p
        p_ = np.asarray([p(x, theta) for x in xs])
        q_ = np.asarray([q(x) for x in xs])
        mask = p_ / (k * q_) > cs
        out.extend(xs[mask])
        i += 1
    
    return out[:size]

def nll0(data, theta):
    p_ = np.asarray([p(x, theta) for x in data])
    return  -np.sum(np.log(p_))

def nll_poiss(data, nu):
    n = len(data)
    return nu - n*np.log(nu)

def nll(data, theta, nu):
    return nll0(data, theta) + nll_poiss(data, nu)

def get_total_counts(theta):
    M = 123800.645  # MeV
    mMu = 105.6583755  # MeV

    max_nuE = mMu/2.
    max_recoil_nr = 2*(max_nuE)**2/M

    light_yield = 13.35 * 1000. # PE/MeV
    max_pe = light_yield * quenching_factor(max_recoil_nr)

    _theta = (ctypes.c_double*2)(theta[0], theta[1])
    user_data = ctypes.cast(_theta, ctypes.c_void_p)

    func = LowLevelCallable(lib.recoil_spectrum_int, user_data)

    flux_average_xs = tplquad(func, 0, max_pe, lambda pe: 0, lambda pe: max_recoil_nr, lambda pe,Er: np.sqrt(M*Er/2), lambda pe,Er: max_nuE)[0]

    # flux_average_xs = tplquad(lambda Enu,Er,pe: EeeFlux(Enu, 19.3, theta[0], theta[1])
    #                                                 * XSection(Enu, Er)
    #                                                 * smear(pe, quenching_factor(Er)),
    #                                                 0, max_pe, lambda pe: 0, lambda pe: max_recoil_nr, lambda pe,Er: np.sqrt(M*Er/2), lambda pe,Er: max_nuE)[0]

    num_atoms = 4.53e24 # 1kg Cs
    flux = 8.46e14 # nu/cm2/yr (at 20m)
    exposure = 135.0 # kg yr

    return flux_average_xs * num_atoms * flux * exposure


def main():
    # True parameters are u_e4^2 = 0.25 and delta m^2 = 1
    true_u = 0.25
    true_m = 1

    n_observed = get_total_counts([true_u, true_m])
    print(n_observed)
    data = sample(int(n_observed), [true_u, true_m])

    # print(data)
    
    # Set up a 2d grid to do some FC on
    u = np.linspace(0, 1, 20)
    m = np.linspace(0, 10, 20)

    n_toy = 1000

    chi2s = 1e6*np.ones((len(u), len(m)))
    chi2_mins = 1e6*np.ones((len(u), len(m)))
    chi2_crits = 1e6*np.ones((len(u), len(m)))

    for i in range(len(u)):
        print(i)
        for j in range(len(m)):
            print(j)
            theta = [u[i], m[j]]
            nu0 = get_total_counts(theta)
            _nll0 = nll0(data, theta)

            chi2s[i,j] = -2 * (_nll0 + nll_poiss(data, nu0))

            delta_nu = np.random.poisson(nu0, n_toy)
            nll_arr = [_nll0 + nll_poiss(data, nu) for nu in delta_nu]

            chi2s_toys = -2 * np.asarray(nll_arr)
            chi2_crits[i,j] = np.percentile(chi2s_toys, 90)
            chi2_mins[i,j] = np.min(chi2s_toys)

    global_min = np.min(chi2s)

    acceptance = np.zeros_like(chi2s)
    for i in range(len(u)):
        for j in range(len(m)):
            delta_chi2_c = chi2_crits[i,j] - chi2_mins[i,j]
            delta_chi2_i = chi2s[i,j] - global_min

            acceptance[i,j] = delta_chi2_i - delta_chi2_c

    
    # Plot the acceptance region
    plt.imshow(acceptance)
    plt.colorbar()
    plt.xlabel("Mass")
    plt.ylabel("Sin^2(2θ)")
    plt.title(f"Acceptance Region for 90 perc")
    plt.show()
            


    # q_ = [q(p_, max_p) for p_ in pe]

    # fig, ax = plt.subplots()
    # ax.plot(pe, 5000*np.asarray(pe_spec)/np.sum(np.asarray(pe_spec)), label=r"$p(x)$")
    # ax.hist(s, bins=50, alpha=0.5, label="Sampled")
    # # ax.plot(pe, q_, label=r"$q(x)$")
    # plt.legend()
    # plt.show()

    return

    pe_spectrum_unosc = lambda pe: dblquad(lambda Enu,Er: EeeFlux(Enu, 19.3, 0, 0)
                                                    * XSection(Enu, Er)
                                                    * smear(pe, quenching_factor(Er)),
                                                    0, max_recoil_nr, lambda Er: np.sqrt(M*Er/2), lambda Er: max_nuE)[0]
    
    pe_spectrum_osc = lambda pe: dblquad(lambda Enu,Er: EeeFlux(Enu, 19.3, 0.5, 1)
                                                    * XSection(Enu, Er)
                                                    * smear(pe, quenching_factor(Er)),
                                                    0, max_recoil_nr, lambda Er: np.sqrt(M*Er/2), lambda Er: max_nuE)[0]

    # letsa plot

    pe = np.linspace(0, 50, 100)
    pe_spec_unosc = [pe_spectrum_unosc(p) for p in pe]
    pe_spec_osc = [pe_spectrum_osc(p) for p in pe]

    plt.plot(pe, pe_spec_osc)
    plt.plot(pe, pe_spec_unosc)
    plt.show()


    # print(er_spectrum(0.002))
    # print(smear(10, 0.008))

if __name__ == "__main__":
    # cProfile.run("main()", "output.prof")
    main()