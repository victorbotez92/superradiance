import numpy as np
from numpy.fft import fft,fftfreq
import matplotlib.pyplot as plt

import os
import time


from read_data import global_parameters
from RK_num_scheme import *
from random_features import *
from fourier_transforms import *


class parameters:
    def __init__(self,list_ints,elms_ints,list_floats,elms_floats,list_bools,elms_bools,elms_chars):
        for i in range(len(list_ints)):
            setattr(self,list_ints[i],elms_ints[i])
        for i in range(len(list_floats)):
            setattr(self,list_floats[i],elms_floats[i])
        for i in range(len(list_bools)):
            setattr(self,list_bools[i],elms_bools[i])
        for i in range(len(list_chars)):
            setattr(self,list_chars[i],elms_chars[i])



#############################################
#############################################
#### Definition of Global Parameters ########
#############################################
#############################################

data_file = 'parameters.txt'
list_ints = ['nb_samples_tau','nb_samples_z','nb_velocities','nb_savings_z','num_velocity_channel']
list_floats = ['tau_max','z_max','z_min','c','e_0','D','h_barre','tau_sp','T1','T2','f0','d','L','n0','v_sep','z_save_min','z_save_max']
list_bools = ['random_phase','random_amplitude','endfire','plot_fourier','transverse_inversion','constant_pumping','save_outputs']
list_chars = ['name_file']


elms_ints,elms_floats,elms_bools,elms_chars = global_parameters(data_file,list_ints,list_floats,list_bools,list_chars)

for i in range(len(list_ints)):
    globals()[list_ints[i]] = elms_ints[i]

for i in range(len(list_floats)):
    globals()[list_floats[i]] = elms_floats[i]

for i in range(len(list_bools)):
    globals()[list_bools[i]] = elms_bools[i]

for i in range(len(list_chars)):
    globals()[list_chars[i]] = elms_chars[i]

##############################################

wavelength = c/f0
w0 = 2*np.pi*f0

if transverse_inversion:
    tau_min = -L/c
else:
    tau_min = 0
dtau = (tau_max-tau_min)/nb_samples_tau
dz = (z_max-z_min)/nb_samples_z

dv = 2*np.pi*c/w0/tau_max
v_sep = v_sep*dv

Tr = tau_sp*(8*np.pi)/(3*wavelength**2*L*n0)  # Superradiant timescale

v_eval = v_sep*np.arange(nb_velocities)   # Relative velocities array
v_eval -= v_eval[-1]/2
w_eval = w0*v_eval/c  # Relative pulsations array

full_w = np.repeat(w_eval,nb_samples_z)

N_tot = n0*wavelength*L**2
theta_0 = np.sqrt(4/N_tot)




all_parameters = parameters(list_ints,elms_ints,list_floats,elms_floats,list_bools,elms_bools,elms_chars)
all_parameters.dz = dz
all_parameters.dtau = dtau
all_parameters.Tr = Tr
all_parameters.full_w = full_w
all_parameters.N_tot = N_tot
all_parameters.tau_min = tau_min

#############################################
#############################################
#############################################
#############################################
#############################################




#############################################
#############################################
######### Usefull functions #################
#############################################
#############################################

def splitting(y):
    return y[:len(y)//2],y[len(y)//2:]


def make_theta_0(all_parameters):
    theta_0_matrix = np.ones(all_parameters.nb_velocities*all_parameters.nb_samples_z,dtype=complex)  # + 2
    if all_parameters.random_phase:
        theta_0_matrix*=build_random_phases(all_parameters)
    if all_parameters.random_amplitude:
        theta_0_matrix*=build_random_amplitudes(all_parameters)
    else:
        theta_0_matrix*=theta_0
    return theta_0_matrix

def add_timestep_to_save(y_cur,E_cur,N_save,P_save,E_save,list_z_save,nb_timestep,all_parameters):
    N_save[nb_timestep] = np.real(y_cur[all_parameters.num_velocity_channel*all_parameters.nb_samples_z+list_z_save])
    P_save[nb_timestep] = y_cur[all_parameters.num_velocity_channel*all_parameters.nb_samples_z+list_z_save+len(y_cur)//2]
    E_save[nb_timestep] = E_cur[list_z_save]

#############################################
#############################################
######### Lists used to make outputs ########
#############################################
#############################################

if endfire:
    list_z_save = np.array([nb_samples_z-1])
else:
    list_z_save_provisory = nb_samples_z/(z_max-z_min)*np.linspace(z_save_min,z_save_max,nb_savings_z)
    list_z_save = np.empty(nb_savings_z,dtype=int)
    for i,elm in enumerate (list_z_save_provisory):
        list_z_save[i] = int(np.floor(elm))
    list_z_save -= 1

N_save = np.empty((nb_samples_tau,nb_savings_z),dtype=float)
P_save = np.empty((nb_samples_tau,nb_savings_z),dtype=complex)
E_save = np.empty((nb_samples_tau,nb_savings_z),dtype=complex)

#############################################
#############################################
############## Initialization ###############
#############################################
#############################################


theta_0_matrix = make_theta_0(all_parameters)
all_parameters.theta_0_matrix = theta_0_matrix

N_ini = np.ones(nb_velocities*nb_samples_z,dtype = complex)
P_ini = theta_0_matrix[:nb_velocities*nb_samples_z]*np.ones(nb_velocities*nb_samples_z,dtype = complex)
y_ini = np.concatenate((N_ini,P_ini))

E_ini = 1.j*theta_0*np.linspace(z_min,z_max,nb_samples_z)


start = time.time()
checkpoint = start
tau = tau_min

y_cur,E_cur = y_ini,E_ini

#############################################
#############################################
################ MAIN LOOP ##################
#############################################
#############################################

iter = 0

while tau < tau_max:
    add_timestep_to_save(y_cur,E_cur,N_save,P_save,E_save,list_z_save,iter,all_parameters)

    if time.time()-checkpoint > 10:
        checkpoint = time.time()
        print(tau)

    if random_phase or random_amplitude:
        theta_0_matrix = make_theta_0(all_parameters)
        all_parameters.theta_0_matrix = theta_0_matrix

    new_y,new_E = t_step(y_cur,E_cur,tau,all_parameters)
    y_cur,E_cur = new_y.copy(),new_E.copy()

    iter += 1
    tau += dtau

print('The simulation of '+str(nb_velocities)+' channels of velocity and separation '+str(int(v_sep/dv))+'dv took '+str((time.time()-start)*100//1/100)+'s')


I_save = np.real(E_save*np.conjugate(E_save))

#########################################
#########################################
############# Make Plots ################
#########################################
#########################################


times = np.linspace(tau_min,tau_max,nb_samples_tau)

if save_outputs:
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, name_file)


plt.figure(0)
plt.clf()
plt.plot(times,N_save)
plt.xlabel('time')
plt.ylabel('inversion population')
plt.title('N')
plt.grid(True)
plt.pause(0.001)

if save_outputs:
    plt.savefig(filename+'_N')


plt.figure(1)
plt.clf()
plt.plot(times,np.abs(P_save))
plt.xlabel('time')
plt.ylabel('polarization')
plt.title('P')
plt.grid(True)
plt.pause(0.001)

if save_outputs:
    plt.savefig(filename+'_P')

plt.figure(2)
plt.clf()
plt.plot(times,I_save)
plt.xlabel('time')
plt.ylabel('intensity')
plt.title('I')
plt.grid(True)
plt.pause(0.001)

if save_outputs:
    plt.savefig(filename+'_I')

# if plot_fourier:
#     freq,I_fft = do_fft(np.linspace(0,tau_max,nb_samples_tau),I_save[-1],all_parameters)
#     plt.figure(3)
#     plt.clf()
#     plt.plot(freq,np.abs(I_fft))
#     plt.xlabel('freq')
#     plt.ylabel('I_fft')
#     plt.title('I_fft')
#     plt.grid(True)
#     plt.pause(0.001)

plt.show() 

if save_outputs:
    np.save(filename+'_N.npy',N_save)
    np.save(filename+'_P.npy',P_save)
    np.save(filename+'_E.npy',E_save)