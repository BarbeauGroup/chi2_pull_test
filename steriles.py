from flux.nuflux import oscillate_flux
from utils.loadparams import load_params
from utils.data_loaders import read_flux_from_root
from flux.create_observables import create_observables
from plotting.observables import plot_observables

def main():
    params = load_params("config/csi.json")
    flux = read_flux_from_root(params)
    oscillated_flux = oscillate_flux(flux=flux)

    un_osc_obs = create_observables(params=params, flux=flux)
    osc_obs = create_observables(params=params, flux=oscillated_flux)

    plot_observables(unosc=un_osc_obs, osc=osc_obs)
    
if __name__ == "__main__":
    main()