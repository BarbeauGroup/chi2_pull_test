import numpy as np

def R0(A):
    r0 = 0.52 # fm
    s = 0.9 # fm    
    return np.sqrt((1.23 * A**(1/3) - 0.60)**2 + (7/3)*np.pi**2 * r0**2 - 5*s**2)

def j1(x):
    return np.sin(x)/(x**2) - np.cos(x)/x

def T_to_q(T, M):
    return np.sqrt(2*M*T + T**2)

def helm(T, M, A):
    q = T_to_q(T, M)
    return (3 * j1(q * R0(A)) / (q * R0(A)))*np.exp(-q**2 * R0(A)**2 / 6)
    
