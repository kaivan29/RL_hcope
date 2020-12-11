import numpy as np

def PDIS_D(D_data, pi_e, gamma = 0.95):   
    pdis_estimates = []
    n = len(D_data)
    
    for i in range(n):
        data = D_data[i]
        timesteps = len(data["S"])

        #Vectorized implementation
        S = np.array(data["S"], dtype=int)
        A = np.array(data["A"], dtype=int)
        R = np.array(data["R"], dtype=float)
        PI_B = np.array(data["PI"], dtype=float)
        PI_E = pi_e.get_probabilities(S, A)

        PI_B[np.where(PI_B == 0)[0]] = 0.00001
        G = np.ones(timesteps)
        G = np.power(gamma, np.arange(timesteps))

        PI_ratio = np.exp(np.cumsum(np.log(PI_E)) - np.cumsum(np.log(PI_B)))

        pdis_t = np.sum(G * PI_ratio * R)
        pdis_estimates.append(pdis_t)

        #print("PDI:", np.mean(np.array(pdis_estimates)))
        
    return pdis_estimates, np.mean(np.array(pdis_estimates))
