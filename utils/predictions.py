import numpy as np

def P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14):
    return 1 - sin_squared_2_theta_14 * np.sin( 1.27*delta_m14_squared * L / E)**2

def toy_model(E, L, delta_m14_squared, sin_squared_2_theta_14, SM_prediction):
    return SM_prediction * P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14)

def flux_dependent_prediction(E, L, delta_m14_squared, sin_squared_2_theta_14, flux_norm_param):
    return flux_norm_param * P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14)