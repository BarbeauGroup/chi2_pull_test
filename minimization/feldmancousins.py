import numpy as np
from itertools import product
import iminuit
from multiprocessing import Pool
from itertools import product
from functools import partial
from tqdm import tqdm
from scipy.optimize import Bounds
from utils import istarmap

def evaluate_gridpoint(i, j, /, ensemble, x0, sin_bins, mass_bins, angle):
    sin_2 = sin_bins[i]
    mass = mass_bins[j]

    if angle == "ee":
        Ue4_2 = 0.5 * (1 - np.sqrt(1 - sin_2))

        bounds_l = np.append([0.], np.full(len(x0[:-1]), -50))
        bounds_u = np.append([0.25], np.full(len(x0[:-1]), 150))
        bounds = Bounds(bounds_l, bounds_u, keep_feasible=True)
        res = iminuit.minimize(ensemble, x0, (mass, Ue4_2), bounds=bounds)

        x_min = res.x

        # Generate predicted histogram given the best fit parameters
        hists = ensemble.analysis_hists(x_min[1:], mass, Ue4_2, x_min[0])

        # Calculate n cost functions for the predicted histogram varied Poisson-like
        costs = []
        for _ in range(1000):
            costs.append(ensemble.cost(x_min[1:], hists_arr=hists, poisson=True))

        return i, j, res.fun, costs, x_min
    
    elif angle == "me":
        # scan over Ue4_2 and calulate Umu4_2 from sin_2
        bounds_l = np.append([sin_2 / 2.], np.full(len(x0[:-1]), -np.inf))
        bounds_u = np.append([0.5], np.full(len(x0[:-1]), np.inf))
        bounds = Bounds(bounds_l, bounds_u, keep_feasible=True)
        res = iminuit.minimize(ensemble, x0, (mass, None, None, sin_2), bounds=bounds)

        samples = ensemble.generate_samples(ensemble.nuisance_params, 1000)
        u_samples = np.random.uniform(sin_2 / 2., 0.5, size=1000)
        costs = []
        for sample, u_sample in zip(samples, u_samples):
            cost = ensemble(sample, mass, u_sample, None, sin_2)
            costs.append(cost)

        # return (i, j, res.fun)
        # print("mass", mass)
        # print("sin2", sin_2)
        # print("Umu4", sin_2 / (4 * res.x[0]))
        return i, j, res.fun, costs

    elif angle == "mm":
        Umu4_2 = 0.5 * (1 - np.sqrt(1 - sin_2))
        bounds = None
        res = iminuit.minimize(ensemble, x0, (mass, None, Umu4_2), bounds=bounds)
        return (i, j, res.fun)

def feldmancousins(ensemble, x0, sin_bins, mass_bins, angle: str, fname):

    # print(ensemble.generate_samples(ensemble.nuisance_params, 1))
    # return
    param_grid = list(product(range(len(sin_bins)), range(len(mass_bins))))

    # At every point, minimize over parameters
    with Pool() as pool:
        results = list(tqdm(pool.istarmap(partial(evaluate_gridpoint, ensemble=ensemble, sin_bins=sin_bins, mass_bins=mass_bins, x0=x0, angle=angle), param_grid), total=len(param_grid)))

    chi2 = np.full((len(sin_bins), len(mass_bins)), 1e6)
    costs = np.full((len(sin_bins), len(mass_bins), 1000), 1e6)
    xs = np.full((len(sin_bins), len(mass_bins), len(x0)), 0)

    for i, j, val, c, x_min in results:
        chi2[i, j] = val
        costs[i, j] = c
        xs[i, j] = x_min

        # Monte Carlo Step

    np.save(f"{fname}_{angle}_chi2_fc.npy", chi2)
    np.save(f"{fname}_{angle}_costs_fc.npy", costs)
    np.save(f"{fname}_{angle}_params_fc.npy", xs)

    # At every point run a monte carlo simulation

    # At every point calculate the delta chi2_c for the given alpha(s)

    # Save whether that point is in the allowed region for the given alpha(s)

    # Save the results (delta chi2, delta chi2_c(s), params, success) for every point