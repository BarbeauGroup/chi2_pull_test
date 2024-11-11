import emcee
import numpy as np
import corner
from multiprocessing import Pool

from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt
from stats.fit import cost_function

from flux.nuflux import oscillate_flux
from flux.create_observables import create_observables
from plotting.observables import analysis_bins, plot_observables


def plot():
    params = load_params("config/csi.json")
    bkd_dict = read_brns_nins_from_txt(params)
    data_dict = read_data_from_txt(params)
    flux = read_flux_from_root(params)

    x = [-5.20975558e-04, 5.56406371e-01, 5.71966714e-03, 4.57742461e-03,
  1.34700734e-01, 5.90428294e-03, 1.36187423e-02, 8.57666835e-03]
    osc_params = x[0:4]
    osc_params = [params["detector"]["distance"]/100., osc_params[0], osc_params[1], osc_params[2], osc_params[3]]

    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)

    un_osc_obs = create_observables(params=params, flux=flux)
    osc_obs = create_observables(params=params, flux=oscillated_flux)

    histograms_unosc = analysis_bins(observable=un_osc_obs, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)
    histograms_osc = analysis_bins(observable=osc_obs, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)

    plot_observables(params, histograms_unosc, histograms_osc, x[-1])

def main():
    params = load_params("config/csi.json")
    bkd_dict = read_brns_nins_from_txt(params)
    data_dict = read_data_from_txt(params)
    flux = read_flux_from_root(params)

    x = [1, 0.3162, 0, 0, 0.28, 0.25, 0.7, 0.04]
    args = (flux, params, bkd_dict, data_dict)

    pos = x + 1e-4 * np.random.randn(30, len(x))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, cost_function, args=args)
    sampler.run_mcmc(pos, 10000, progress=True)

    # print the best fit values
    samples = sampler.get_chain()
    # print(samples)
    flat_samples = sampler.get_chain(discard=200, flat=True)

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{mcmc[1]:.5f} +{q[1]:.5f} -{q[0]:.5f}")
    
    # corner plots
    fig = corner.corner(flat_samples, labels=["deltam41", "Ue4", "Umu4", "Utau4", "a_flux", "a_brn", "a_nin", "a_ssb"])
    fig.savefig("corner.png")


if __name__ == "__main__":
    main()
    # plot()