import numpy as np


def splitting(y):
    return y[:len(y)//2],y[len(y)//2:]

def prop_z(P,all_parameters):   # Computation of electric field
    
    summed_P = 1.j/all_parameters.nb_velocities*np.conjugate(np.sum(P.reshape(all_parameters.nb_velocities,all_parameters.nb_samples_z),axis = 0))
    
    cumsummed_P = np.cumsum(summed_P)
    
    fixed_P = np.cumsum(summed_P[1:])
    retarded_P = np.cumsum(summed_P[:len(summed_P)-1])
    
    E = np.concatenate((np.array([0+0.j]),all_parameters.dz/2*(cumsummed_P[:len(cumsummed_P)-1]+cumsummed_P[1:]-summed_P[0])))
        
    return E

def t_step(y_cur,E_cur,tau,all_parameters,comm,rank,size): #NEW
    
    y_rk1,E_rk1 = rk_t_term(E_cur,y_cur,y_cur,tau,all_parameters,comm,rank,size)
    y_rk2,E_rk2 = rk_t_term(1/2*(E_cur+E_rk1),1/2*(y_cur+y_rk1),y_cur,tau+all_parameters.dtau/2,all_parameters,comm,rank,size)
    y_rk3,E_rk3 = rk_t_term(1/2*(E_cur+E_rk2),1/2*(y_cur+y_rk2),y_cur,tau+all_parameters.dtau/2,all_parameters,comm,rank,size)
    y_rk4,E_rk4 = rk_t_term(E_rk3,y_rk3,y_cur,tau+all_parameters.dtau,all_parameters,comm,rank,size)
    
    new_E = 1/6*(E_rk1+2*E_rk2+2*E_rk3+E_rk4)
    new_y = 1/6*(y_rk1+2*y_rk2+2*y_rk3+y_rk4)
    
    return new_y,new_E

def rk_t_term(E_dpt,y_dpt,y_prev,tau,all_parameters,comm,rank,size):

    full_E_dpt = np.resize(E_dpt,all_parameters.nb_velocities*all_parameters.nb_samples_z)

    if all_parameters.transverse_inversion:
        pump = tau > -np.linspace(all_parameters.z_min,all_parameters.z_max,int(all_parameters.nb_samples_z))*all_parameters.L/all_parameters.c
    else:
        pump = 1
    pump *= all_parameters.constant_pumping

    N_prev,P_prev = splitting(y_prev)
    N_dpt,P_dpt = splitting(y_dpt)  
    
    N_rk = N_prev - all_parameters.dtau*np.imag(P_dpt*full_E_dpt)/all_parameters.Tr-all_parameters.dtau*(N_dpt-pump)/all_parameters.T1
    # theta_and_phase = all_parameters.theta_0_matrix[int((tau-all_parameters.tau_min)/all_parameters.dtau)]*np.exp(1.j*all_parameters.full_w*tau)
    theta_and_phase = all_parameters.theta_0_matrix*np.exp(1.j*all_parameters.full_w*tau)
    P_rk = P_prev + all_parameters.dtau*1.j*N_dpt*np.conjugate(full_E_dpt)/all_parameters.Tr-all_parameters.dtau*(P_dpt-theta_and_phase)/all_parameters.T2+all_parameters.dtau*1.j*all_parameters.full_w*P_dpt
    
    y_rk = np.concatenate((N_rk,P_rk))
    E_rk = prop_z(P_rk,all_parameters)

    if rank != 0 and size > 1: # NEW
        E_to_add = comm.recv(source=rank-1)
        E_rk += E_to_add
        # print(f'rank {rank} received from rank {rank-1}')
    if rank != size -1 and size > 1:
        comm.send(E_rk[-1],dest=rank+1)
        # print(f'rank {rank} sent to rank {rank+1}')
    return y_rk,E_rk


# def t_step_transverse(y_cur,E_cur,tau,all_parameters): #cur stands for current
    
#     y_rk1,E_rk1 = rk_t_term_transverse(E_cur,y_cur,y_cur,tau,all_parameters)
#     y_rk2,E_rk2 = rk_t_term_transverse(1/2*(E_cur+E_rk1),1/2*(y_cur+y_rk1),y_cur,tau+all_parameters.dtau/2,all_parameters)
#     y_rk3,E_rk3 = rk_t_term_transverse(1/2*(E_cur+E_rk2),1/2*(y_cur+y_rk2),y_cur,tau+all_parameters.dtau/2,all_parameters)
#     y_rk4,E_rk4 = rk_t_term_transverse(E_rk3,y_rk3,y_cur,tau+all_parameters.dtau,all_parameters)
    
#     new_E = 1/6*(E_rk1+2*E_rk2+2*E_rk3+E_rk4)
#     new_y = 1/6*(y_rk1+2*y_rk2+2*y_rk3+y_rk4)
    
#     return new_y,new_E

# def rk_t_term_transverse(E_dpt,y_dpt,y_prev,tau,all_parameters):
    
    
#     transverse_pump = tau > -all_parameters.z_eval*all_parameters.L/all_parameters.c
    
    
#     full_E_dpt = np.resize(E_dpt,all_parameters.nb_velocities*all_parameters.nb_samples_z)
    
#     N_prev,P_prev = splitting(y_prev)
#     N_dpt,P_dpt = splitting(y_dpt)  
    
#     N_rk = N_prev - all_parameters.dtau*np.imag(P_dpt*full_E_dpt)/all_parameters.Tr-all_parameters.dtau*(N_dpt-transverse_pump)/all_parameters.T1
#     theta_and_phase = all_parameters.theta_0_matrix[int(tau/all_parameters.dtau)]*np.exp(1.j*all_parameters.full_w*tau)
#     P_rk = P_prev + all_parameters.dtau*1.j*N_dpt*np.conjugate(full_E_dpt)/all_parameters.Tr-all_parameters.dtau*(P_dpt-theta_and_phase)/all_parameters.T2+all_parameters.dtau*1.j*all_parameters.full_w*P_dpt
    
#     y_rk = np.concatenate((N_rk,P_rk))
#     E_rk = prop_z(P_rk,all_parameters)
    
#     return y_rk,E_rk