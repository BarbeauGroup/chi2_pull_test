from flux.nuflux import oscillate_flux
from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root, read_brns_nins_from_txt, read_data_from_txt
from flux.create_observables import create_observables
from plotting.observables import plot_observables, analysis_bins
from stats.chi2 import chi2_stat

def main():
    params = load_params("config/csi.json")

    bkd_dict = read_brns_nins_from_txt(params)

    data_dict = read_data_from_txt(params)

    # print(data_dict)
    # return
    
    flux = read_flux_from_root(params)

    # Define flux parameters
    osc_params = {}
    osc_params["L"] = 19.3
    osc_params["deltam41"] = 1
    osc_params["Ue4"] = 0.3162
    osc_params["Umu4"] = 0
    osc_params["Utau4"] = 0.0


    oscillated_flux = oscillate_flux(flux=flux, oscillation_params=osc_params)

    un_osc_obs = create_observables(params=params, flux=flux)
    osc_obs = create_observables(params=params, flux=oscillated_flux)

    histograms_unosc = analysis_bins(observable=un_osc_obs, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)
    histograms_osc = analysis_bins(observable=osc_obs, bkd_dict=bkd_dict, data=data_dict, params=params, brn_norm=18.4, nin_norm=5.6)

    chi2 = chi2_stat(histograms=histograms_osc)

    print(chi2)

    # plot_observables(histograms_osc=histograms_osc, histograms_unosc=histograms_unosc, params=params)
    
if __name__ == "__main__":
    main()