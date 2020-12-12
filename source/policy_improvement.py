import time
import numpy as np
from pdis import compute_PDIS
from scipy import stats
import sys
import cma
from softmax import Softmax


class PolicyImprovement:
    
    def __init__(self, D, train_c, test_s, theta_b, policy_e, delta:int, c:int, gamma:int):
        self.D = D
        self.train_c = train_c
        self.test_s = test_s
        self.test_s_size = len(test_s)
        self.theta_b = theta_b
        self.policy_e = policy_e
        self.policy_c = None
        self.c = c
        self.delta = delta
        self.gamma = gamma
        
    def flatten_theta_b(self):
        return self.theta_b.flatten()
    
    def evaluate(self, sigma):
        start_time = time.time()
        
        CMAES = cma.CMAEvolutionStrategy(self.flatten_theta_b, sigma, {'maxiter':10, 'popsize': 10, 'seed': 111})
        possible_theta, evaluation_results = CMAES.ask_and_eval(self.objectiveFunction)
        min_index = np.argmin(evaluation_results)
        
        print(f'Total {len(evaluation_results)} selected thetas were computed in: {time.time() - start_time}')
        print("CMAE evaluation results: ", evaluation_results)
        
        CMAES.tell(possible_theta, evaluation_results)
        CMAES.logger.add()
        CMAES.disp()
        #print(CMAES.result_pretty())
        selected_theta = possible_theta[min_index]
        
        return selected_theta, evaluation_results[min_index]
        
    
    def objectiveFunction(self, theta_e):
        self.policy_e.parameters = theta_e
        # generate pdis
        PDIS_est, PDIS_d = compute_PDIS(self.train_c, self.policy_e, self.gamma)
        print("Mean pdis:", PDIS_d)
        
        variance_c = np.std(PDIS_est)
        print("Theta variance: ", variance_c)
        
        var = (2 * variance_c/np.sqrt(self.test_s_size))
        stat = stats.t.ppf(1-self.delta, self.test_s_size-1)
        safety_term = var * stat
        safetyVal = PDIS_d - safety_term
        
        threshold_pass = (safetyVal >= self.c)
        print("Threshold pass: ", threshold_pass)
        
        if(threshold_pass):
            return -PDIS_d
        else:
            return 1000000    
    
    def safetyTest(self, theta_c, policy_c):
        self.policy_c = policy_c
        self.policy_c.parameters = theta_c
        # generate pdis
        PDIS_est, PDIS_d = compute_PDIS(self.test_s, self.policy_c, self.gamma)
        print("Mean pdis:", PDIS_d)
        
        variance_s = np.std(PDIS_est)
        print("Theta variance:", variance_s)
        
        var = (variance_s/np.sqrt(self.test_s_size))
        stat = stats.t.ppf(1-self.delta, self.test_s_size-1)
        safety_term = var * stat
        safetyVal = PDIS_d - safety_term
        
        if(safetyVal>=self.c):
            return True, PDIS_d
        else:
            return False, PDIS_d