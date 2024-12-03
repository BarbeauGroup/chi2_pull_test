import numpy as np
import iminuit
from scipy.optimize import Bounds
from multiprocessing import Pool
from itertools import product
from functools import partial
from tqdm import tqdm

import sys
sys.path.append("../")
from utils import istarmap


def marginalize_mass_u_helper(i, j, /, cost, x0, u_bins, mass_bins):
    u = u_bins[i]
    mass = mass_bins[j]
    bounds = Bounds(np.append(np.full(len(x0[:-1]), -np.inf), 0), np.append(np.full(len(x0[:-1]), np.inf), 1-u), keep_feasible=True)
    res = iminuit.minimize(cost, x0, args=(mass, u), bounds=bounds)
    return (i, j, res.fun, res.x[-1], res.success, res.x)

def marginalize_mass_u(cost, x0, u_bins, mass_bins, fname):
    param_grid = list(product(range(len(u_bins)), range(len(mass_bins))))
                      
    with Pool() as pool:
        results = list(tqdm(pool.istarmap(partial(marginalize_mass_u_helper, cost=cost, u_bins=u_bins, mass_bins=mass_bins, x0=x0), param_grid), total=len(param_grid)))

    chi2 = np.full((len(u_bins), len(mass_bins)), 1e6)
    margin_u = np.zeros((len(u_bins), len(mass_bins)))
    success = np.zeros((len(u_bins), len(mass_bins)))
    xs = np.zeros((len(u_bins), len(mass_bins), len(x0)))

    for i, j, val, u, suc, x in results:
        chi2[i, j] = val
        margin_u[i, j] = u
        success[i, j] = suc
        xs[i, j] = x

    chi2 = chi2 - np.min(chi2)
    np.save(f"{fname}_chi2_mass_u.npy", chi2)
    np.save(f"{fname}_marginu_mass_u.npy", margin_u)
    np.save(f"{fname}_success_mass_u.npy", success)
    np.save(f"{fname}_params_mass_u.npy", xs)

def marginalize_mass_uu_helper(i, j, k, /, cost, x0, u_bins, mass_bins):
    u1 = u_bins[i]
    u2 = u_bins[j]
    mass = mass_bins[k]
    if u1 + u2 > 1:
        return (i, j, k, 1e6, np.zeros_like(x0), False)
    res = iminuit.minimize(cost, x0, args=(mass, u1, u2))
    return (i, j, k, res.fun, res.x, res.success)

def marginalize_mass_uu(cost, x0, u_bins, mass_bins, fname):
    param_grid = list(filter(lambda x: u_bins[x[0]] + u_bins[x[1]] <= 1,
        product(range(len(u_bins)), range(len(u_bins)), range(len(mass_bins)))))

    with Pool() as pool:
        results = list(tqdm(pool.istarmap(partial(marginalize_mass_uu_helper, cost=cost, u_bins=u_bins, mass_bins=mass_bins, x0=x0), param_grid), total=len(param_grid)))
                       
    chi2 = np.full((len(u_bins), len(u_bins), len(mass_bins)), 1e6)
    xs = np.zeros((len(u_bins), len(u_bins), len(mass_bins), len(x0)))
    success = np.zeros((len(u_bins), len(u_bins), len(mass_bins)))

    for i, j, k, val, x, suc in results:
        chi2[i, j, k] = val
        xs[i, j, k] = x
        success[i, j, k] = suc
        
    chi2 = chi2 - np.min(chi2)
    np.save(f"{fname}_chi2_mass_uu.npy", chi2)
    np.save(f"{fname}_success_mass_uu.npy", success)
    np.save(f"{fname}_params_mass_uu.npy", xs)