import numpy as np

def build_random_phases(all_parameters):
    return np.exp(2*np.pi*1.j*np.random.rand(all_parameters.nb_velocities*all_parameters.nb_samples_z+2))

def build_random_amplitudes(all_parameters):
    random_term = (np.random.exponential(scale = np.sqrt(all_parameters.N_tot),size = all_parameters.nb_velocities*all_parameters.nb_samples_z+2))
    amplitudes = 2/all_parameters.N_tot*random_term
    return amplitudes