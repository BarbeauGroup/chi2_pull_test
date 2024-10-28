import numpy as np


class Experiment:
    def __init__(self, name):
        self.name = name

    def set_distance(self, distance):
        self.distance = distance

    def set_mass(self, mass):
        self.mass = mass

    def set_n_observed(self, n_observed):
        self.n_observed = n_observed
    
    def set_steady_state_background(self, steady_state_background):
        self.steady_state_background = steady_state_background
    
    def set_number_background_windows(self, number_background_windows):
        self.number_background_windows = number_background_windows

    def set_systematic_error_dict(self, systematic_error_dict):
        self.systematic_error_dict = systematic_error_dict

    def set_flux_transfer_matrix(self, flux_transfer_matrix):
        self.flux_transfer_matrix = flux_transfer_matrix

    def set_exposure(self, exposure):
        self.exposure = exposure


    def get_distance(self):
        return self.distance

    def get_name(self):
        return self.name
    
    def get_n_observed(self):
        return self.n_observed
    
    def get_steady_state_background(self):
        return self.steady_state_background
    
    def get_number_background_windows(self):
        return self.number_background_windows
    
    def get_systematic_error_dict(self):
        return self.systematic_error_dict
    
    def get_flux_transfer_matrix(self):
        return self.flux_transfer_matrix
    
    def get_mass(self):
        return self.mass