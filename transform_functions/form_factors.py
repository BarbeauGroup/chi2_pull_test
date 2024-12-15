import numpy as np

r0 = 0.52 # fm
s = 0.9 # fm

"""
A: mass number
returns R0 in fm
"""
def R0(A):    
    return 1.23 * A**(1/3.)# - 0.60)**2 + (7/3.)*np.pi**2 * r0**2 - 5*s**2)

"""
x: unitless
returns j1(x)
"""
def j1(x):
    return np.sin(x)/(x**2) - np.cos(x)/x

"""
T: keV
M: mass in keV/c^2
returns q in 1/fm
"""
def T_to_q(T, M):
    hbarc = 197327. # keV fm
    return np.sqrt(2*M*T + T**2)/hbarc

"""
params: dict
T: keV
alpha: unitless (form factor correction)
returns form factor in unitless
"""
def helm(isotope, T, alpha):
    Da = 931494.1037229 # keV/c^2 / amu

    A = isotope["A"]
    M = isotope["mass"] # amu
    
    M *= Da # keV/c^2

    A *= 1 + alpha

    q = T_to_q(T, M) # 1/fm
    return np.where(q > 0, (3 * j1(q * R0(A)) / (q * R0(A))) * np.exp(-q**2 * s**2 / 2.), 1.0) # unitless
    
