import time
import numpy as np
from pdis import PDIS_D
from scipy import stats
import sys
import cma
from softmax import Softmax


class PolicyImprovement:
    
    def __init__(self, D, train_c, test_s, theta_b, policy_e, delta:int, c:int, gamma:int):
        self._D = D
        self._train_c = train_c
        self._test_s = test_s
        self._test_s_size = len(test_s)
        self._c = c
        self._delta = delta
        self._theta_b = theta_b
        self._policy_e = policy_e
        self._policy_c = None
        self._gamma = gamma
        
    @property
    def theta_b(self)->np.ndarray:
        return self._theta_b.flatten()
    
    @theta_b.setter
    def theta_b(self, p:np.ndarray):
        self._theta_b = p.reshape(self._theta_b.shape)
    
    def evaluate(self,sigma):
        start_time = time.time()
        
        CMAES = cma.CMAEvolutionStrategy(self.theta_b,sigma,{'maxiter':10, 'popsize': 10, 'seed': 111})
        possible_theta, evaluation_results = CMAES.ask_and_eval(self.objectiveFunction)
        min_index = np.argmin(evaluation_results)
        
        print(f'Time to find {len(evaluation_results)} selected thetas: {time.time() - start_time}')
        print("Eval results: ", evaluation_results)
        
        CMAES.tell(possible_theta, evaluation_results)
        CMAES.logger.add()
        CMAES.disp()
        #print(CMAES.result_pretty())
        selected_theta = possible_theta[min_index]
        
        return selected_theta, evaluation_results[min_index]
        
    
    def objectiveFunction(self, theta_e):
        self._policy_e.parameters = theta_e
        # generate pdis
        pdis_d_est,pdis_d = PDIS_D(self._train_c, self._policy_e, self._gamma)
        
        variance_c = np.std(pdis_d_est)
        safety_term = (2 * variance_c/np.sqrt(self._test_s_size)) * stats.t.ppf(1-self._delta, self._test_s_size-1)
        
        print("Theta variance: ", variance_c)
        print(f'Mean pdis: {pdis_d}')
        
        diff = pdis_d - safety_term
        threshold_pass = (diff >= self._c)
        print("Threshold pass: ", threshold_pass)
        
        if(threshold_pass):
            return -pdis_d
        else:
            return 1000000    
    
    def safetyTest(self, theta_c, policy_c):
        self._policy_c = policy_c
        self._policy_c.parameters = theta_c

        pdis_d_est,pdis_d = PDIS_D(self._test_s, self._policy_c, self._gamma)
        sigma_s = np.std(pdis_d_est)
        safety_term = (sigma_s/(np.sqrt(self._test_s_size)))*stats.t.ppf(1-self._delta,self._test_s_size-1)
        
        print("mean:",pdis_d)
        print("variance:",sigma_s)
        
        safetyVal = pdis_d - safety_term
        
        if(safetyVal>=self._c):
            return True, pdis_d
        else:
            return False, pdis_d