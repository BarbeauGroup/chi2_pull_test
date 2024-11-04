import numpy as np


def P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14):
    return 1 - sin_squared_2_theta_14 * np.sin( 1.27*delta_m14_squared * L / E)**2

