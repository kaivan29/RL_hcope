import numpy as np

def compute_PDIS(Data, pi_e, gamma = 0.95):   
    PDIS_estimates = []
    n = len(Data)
    
    for i in range(n):
        data = Data[i]
        timesteps = len(data["S"])

        #Vectorized implementation
        S = np.array(data["S"], dtype=int)
        A = np.array(data["A"], dtype=int)
        R = np.array(data["R"], dtype=float)
        PI_B = np.array(data["PI"], dtype=float)
        PI_E = pi_e.get_probabilities(S, A)

        PI_B[np.where(PI_B == 0)[0]] = 1/100000
        PI_E_B = np.exp(np.cumsum(np.log(PI_E)) - np.cumsum(np.log(PI_B)))
        G = np.power(gamma, np.arange(timesteps))
        
        PDIS_timestep = np.sum(G * PI_E_B * R)
        PDIS_estimates.append(PDIS_timestep)

        #print("PDI:", np.mean(np.array(PDIS_estimates)))
    PDIS_avg = np.mean(np.array(PDIS_estimates))
    
    return PDIS_estimates, PDIS_avg
