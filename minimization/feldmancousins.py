import numpy as np
from itertools import product
import iminuit
from multiprocessing import Pool
from itertools import product
from functools import partial
from tqdm import tqdm
from scipy.optimize import Bounds

def evaluate_gridpoint(i, j, /, cost, x0, sin_bins, mass_bins, angle):
    sin_2 = sin_bins[i]
    mass = mass_bins[j]

    if angle == "ee":
        Ue4_2 = 0.5 * (1 - np.sqrt(1 - sin_2))
        res = iminuit.minimize(cost, x0, (mass, Ue4_2), bounds=bounds)
        return (i, j, res.fun)
    
    elif angle == "me":
        bounds_l = np.append([sin_2 / 2.], np.full(len(x0[:-1]), -np.inf))
        bounds_u = np.append([0.5], np.full(len(x0[:-1]), np.inf))
        bounds = Bounds(bounds_l, bounds_u, keep_feasible=True)
        res = iminuit.minimize(cost, x0, (mass, None, None, sin_2), bounds=bounds)
        # return (i, j, res.fun)
        print("mass", mass)
        print("sin2", sin_2)
        print("Umu4", sin_2 / (4 * res.x[0]))
        return res

    elif angle == "mm":
        Umu4_2 = 0.5 * (1 - np.sqrt(1 - sin_2))
        res = iminuit.minimize(cost, x0, (mass, None, Umu4_2), bounds=bounds)
        return (i, j, res.fun)

def feldmancousins(cost, x0, sin_bins, mass_bins, angle: str, fname):
    param_grid = list(product(range(len(sin_bins)), range(len(mass_bins))))

    # At every point, minimize over parameters
    with Pool() as pool:
        results = list(tqdm(pool.istarmap(partial(evaluate_gridpoint, cost=cost, sin_bins=sin_bins, mass_bins=mass_bins, x0=x0, bounds=bounds, angle=angle), param_grid), total=len(param_grid)))

    chi2 = np.full((len(sin_bins), len(mass_bins)), 1e6)

    for i, j, val in results:
        chi2[i, j] = val
    
    chi2 = chi2 - np.min(chi2)

    np.save(f"{fname}_{angle}_chi2_fc.npy", chi2)

    # At every point run a monte carlo simulation

    # At every point calculate the delta chi2_c for the given alpha(s)

    # Save whether that point is in the allowed region for the given alpha(s)

    # Save the results (delta chi2, delta chi2_c(s), params, success) for every point