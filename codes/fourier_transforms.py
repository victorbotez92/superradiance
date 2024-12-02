import numpy as np
from numpy.fft import fft,fftfreq

def fft_sort(list_to_sort):
    return np.concatenate((list_to_sort[len(list_to_sort)//2:],list_to_sort[:len(list_to_sort)//2]))


def do_fft(t_eval,y,all_parameters):
    N_fourier = len(y)
    L_fourier = all_parameters.tau_max
    # freq = fft_sort(fftfreq(N_fourier,L_fourier/N_fourier))
    # y_fft = fft_sort(fft(y))
    freq = fftfreq(N_fourier,L_fourier/N_fourier)
    y_fft = fft(y)
    return freq,y_fft