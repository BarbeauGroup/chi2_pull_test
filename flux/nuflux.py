

class NeutrinoFlux:
    def __init__(self, filename):
        self.filename = filename

    def set_electron_neutrino_flux(self, flux):
        self.electron_neutrino_flux = flux
    
    def set_muon_neutrino_flux(self, flux):
        self.muon_neutrino_flux = flux

    def set_anti_muon_neutrino_flux(self, flux):
        self.anti_muon_neutrino_flux = flux

