import emcee
import numpy as np
import corner
from multiprocessing import Pool

from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from flux.ssb_pdf import make_ssb_pdf
from plotting.observables import analysis_bins, plot_observables
from stats.likelihood import loglike_stat, loglike_sys

import matplotlib.pyplot as plt

# Maybe needed bc of numpy matrix multiplication
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Global variables for data
params = load_params("config/csi.json")

pkl = True
new_time_edges = np.arange(0, 15125, 125)
if not pkl or not os.path.exists("flux/flux_dict.pkl"):
    flux = read_flux_from_root(params, new_time_edges)
    with open("flux/flux_dict.pkl", "wb") as f:
        np.save(f, flux)
else:
    with open("flux/flux_dict.pkl", "rb") as f:
        flux = np.load(f, allow_pickle=True).item()
    if(flux["nuE"][0][0] != new_time_edges).all():
        flux = read_flux_from_root(params, new_time_edges)
        with open("flux/flux_dict.pkl", "wb") as f:
            np.save(f, flux)

ssb_dict = make_ssb_pdf(params)
bkd_dict = read_brns_nins_from_txt(params)
data_dict = read_data_from_txt(params)

observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
t_bin_arr = np.asarray(params["analysis"]["time_bins"])
flux_matrix = np.load(params["detector"]["flux_matrix"])
detector_matrix = np.load(params["detector"]["detector_matrix"])

def cost_function_global(x: np.ndarray) -> float:
    """
    # x: (deltam41, Ue4, Umu4, Utau4, a_flux, a_brn, a_nin, a_ssb)
    x: (deltam41, Ue4, Umu4, a_flux, a_brn, a_nin, a_ssb)

    returns: negative chi2 value
    """
    osc_params = x[0:3]

    if(osc_params[0] > 100):
        return -np.inf

    if(osc_params[0] < 0 or osc_params[1] < 0 or osc_params[2] < 0):
        return -np.inf
    
    if(np.sum(osc_params[1:3]) > 1):
        return -np.inf

    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], 0.0] #osc_params[3]]   
    nuisance_params = x[3:-1]
    time_offset = x[-1]
    nuisance_param_priors = [params["detector"]["systematics"]["flux"],
                            params["detector"]["systematics"]["brn"],
                            params["detector"]["systematics"]["nin"],
                            params["detector"]["systematics"]["ssb"]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)
    osc_obs = create_observables(params=params, flux=oscillated_flux, time_offset=time_offset, flux_matrix=flux_matrix, detector_matrix=detector_matrix, flavorblind=True)
    histograms_osc = analysis_bins(observable=osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=time_offset)
    
    return loglike_stat(histograms_osc, nuisance_params) + loglike_sys(nuisance_params, nuisance_param_priors)


def plot():
    use_backend = True
    if use_backend:
        x = []
        sampler = emcee.backends.HDFBackend("backend.h5")
        flat_samples = sampler.get_chain(discard=100, flat=True)
        for i in range(8):
            mcmc = np.percentile(flat_samples[:, i], 50)
            x.append(mcmc)
    else:
        x = [1.521e+00, 4.174e-01, 4.462e-01, 1.196e-02, -4.793e-03,
                 -4.465e-03, -1.447e-02, 58.0]
    
    print(x)
    osc_params = x[0:3]
    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], 0.0] #osc_params[3]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)

    un_osc_obs = create_observables(params=params, flux=flux, time_offset=x[-1], flux_matrix=flux_matrix, detector_matrix=detector_matrix)
    osc_obs = create_observables(params=params, flux=oscillated_flux, time_offset=x[-1], flux_matrix=flux_matrix, detector_matrix=detector_matrix)

    histograms_unosc = analysis_bins(observable=un_osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=x[-1])
    histograms_osc = analysis_bins(observable=osc_obs, ssb_dict=ssb_dict, bkd_dict=bkd_dict, data=data_dict, params=params, ssb_norm=1286, brn_norm=18.4, nin_norm=5.6, time_offset=x[-1])

    print(cost_function_global(x))

    plot_observables(params, histograms_unosc, histograms_osc, x[3:-1])

def fit():
    global flux, params, bkd_dict, data_dict  # Declare globals for data

    x = [1.32, np.sqrt(0.116), np.sqrt(0.135), 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial guess

    pos = x + [1., 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 10] * np.random.randn(32, len(x))
    if np.any(pos[:, 0:3] < 0):
        pos[:, 0:3] = np.abs(pos[:, 0:3])
    js = np.where(pos[:, 1] + pos[:, 2] > 1)[0]
    if js.size > 0:
        pos[js, 1] /= 2
        pos[js, 2] /= 2
        
    nwalkers, ndim = pos.shape

    use_backend = False
    if use_backend:
        sampler = emcee.backends.HDFBackend("backend.h5")

    else:
        backend = emcee.backends.HDFBackend("backend.h5")
        backend.reset(nwalkers, ndim)

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, cost_function_global, pool=pool, backend=backend)#, moves=emcee.moves.StretchMove(a=2.0))
            sampler.run_mcmc(pos, 10000, progress=True)

    # print the best fit values
    flat_samples = sampler.get_chain(discard=500, flat=True)

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{mcmc[1]:.5f} +{q[1]:.5f} -{q[0]:.5f}")
    
    # corner plots
    fig = corner.corner(flat_samples, labels=[r"$\Delta m_{41}$", r"$|U_{e4}|^2$", r"$|U_{\mu 4}|^2$", r"$\alpha_{flux}$", r"$\alpha_{brn}$", r"$\alpha_{nin}$", r"$\alpha_{ssb}$", r"$\Delta t$"])
    fig.savefig("corner.png")

    # tau = sampler.get_autocorr_time()
    # print(tau)

if __name__ == "__main__":
    # fit()
    plot()
    # cProfile.run("plot()", "output.prof")