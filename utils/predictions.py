import numpy as np

def P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14):
    return 1 - sin_squared_2_theta_14 * np.sin( 1.27*delta_m14_squared * L / E)**2

def sns_nue_spectrum(E):

    m_mu = 105 # MeV

    f = 96 * E**2 * m_mu**-4 * (m_mu - 2*E)
    f[f < 0] = 0
    return f 

def sns_numu_spectrum(E):

    m_mu = 105 # MeV
    
    return 16. * pow(E, 2) * pow(m_mu,4) * (3*m_mu - 4*E)

def toy_model(E, L, delta_m14_squared, sin_squared_2_theta_14, SM_prediction):
    return SM_prediction * P_nue_nue(E, L, delta_m14_squared, sin_squared_2_theta_14)

def truth_level_prediction(experiment, delta_m14_squared, sin_squared_2_theta_14, verbose = False):

    # Define SNS Flux
    energy_arr =  np.arange(1, 55, 1)
    flux_transfer_matrix = experiment.get_flux_transfer_matrix()
    L = experiment.get_distance()

    flux_transfer_matrix = flux_transfer_matrix[:len(energy_arr), :len(energy_arr)]

    # check that the energy array is the same length as the flux transfer matrix
    if len(energy_arr) != len(flux_transfer_matrix):
        raise ValueError("Energy array and flux transfer matrix must have the same length")
    

    # Oscillate the flux
    osc_flux = sns_nue_spectrum(energy_arr) * P_nue_nue(energy_arr, L, delta_m14_squared, sin_squared_2_theta_14)
    # Do the matrix multiplication
    em_spectrum = np.dot(flux_transfer_matrix, osc_flux)

    # Integrate the differential xs
    flux_avgd_cross_section = np.sum(em_spectrum)*(energy_arr[1] - energy_arr[0])

    flux_avgd_cross_section *= 1.0 # g_A fudge factor

    # 2.906×10^24 atoms of lead in 1 kg
    mass = experiment.get_mass()
    n_pb = 2.906e24 * mass

    # 8.46 × 10^14 ν cm−2 yr−1 at 20 m (flux paper)
    integrated_flux = 8.46e14 * (1/3) * 0.689

    # rescale by distance
    integrated_flux = integrated_flux * (20**2 / L**2)

    integrated_flux *= experiment.exposure

    if verbose:
        print("Running detector: ", experiment.get_name(), " at distance: ", L)
        print("\tThese should be nominal values: ")
        print("\t\tFlux avgd cross section: ", flux_avgd_cross_section)
        print("\t\tN pb: ", n_pb)
        print("\t\tIntegrated flux: ", integrated_flux)
        print("\t\tCounts: ", flux_avgd_cross_section * n_pb * integrated_flux)

    counts = flux_avgd_cross_section * n_pb * integrated_flux

    return counts