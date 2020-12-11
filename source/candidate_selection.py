# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:31:22 2019

@author: Saurabh
"""
import time

import numpy as np
from pdis import PDIS_D
from scipy import stats
import sys
import cma
from tabular_softmax import TabularSoftmax


class CandidateSelection:
    
    def __init__(self, D, train_c, train_s,theta_b, policy_e, delta:int, cRatio:int,gamma:int , maxIter:int=10):
        self._D = D
        self._train_c = train_c
        self._train_s = train_s
        self._train_s_size = len(train_s)
        self._c = cRatio
        self._delta = delta
        self._theta_b = theta_b
        
        self._policy_e = policy_e
        self._policy_c = None
        
        self._gamma = gamma
        self._maxIter = maxIter
        self._variance = 0
        
    @property
    def theta_b(self)->np.ndarray:
        return self._theta_b.flatten()
    
    @theta_b.setter
    def theta_b(self, p:np.ndarray):
        self._theta_b = p.reshape(self._theta_b.shape)
    
    def evaluateCMAES(self,sigma):
        start_time = time.time()
        
        cmaEs = cma.CMAEvolutionStrategy(self.theta_b,sigma,{'maxiter':self._maxIter, 'popsize': 5})
        # call the objectiveFn
        possible_theta, evaluation_results = cmaEs.ask_and_eval(self.objectiveFn)
        
        min_index = np.argmin(evaluation_results)
        print(f'Time to find {len(evaluation_results)} candidate thetas: {time.time() - start_time}')
        print("evaluation_results: " + str(evaluation_results))
        
        cmaEs.tell(possible_theta, evaluation_results)
        cmaEs.logger.add()
        cmaEs.disp()

        print(cmaEs.result_pretty())
        
        candidate_theta = possible_theta[min_index]
        
        return candidate_theta, evaluation_results[min_index]
        
    
    def objectiveFn(self, theta_e):
        self._policy_e.parameters = theta_e
        pdis_d_arr,pdis_d = PDIS_D(self._train_c, self._policy_e, self._gamma)
        
        sigma_c = np.std(pdis_d_arr)
        safety_term = (2*sigma_c/np.sqrt(self._train_s_size))*stats.t.ppf(1-self._delta,self._train_s_size-1)
        
        print("candidate theta variance: ", sigma_c)
        print(f'mean pdis: {pdis_d}')
        
        barierFnVal = pdis_d - safety_term
        
        
        candidate_pass = (barierFnVal >= self._c)
        print("candidate_pass: ", candidate_pass)
        
        if(barierFnVal >= self._c):
            return -pdis_d
        else:
            return -pdis_d+1000000    
    
    def safetyTest(self, theta_c, policy_c):
        self._policy_c = policy_c
        self._policy_c.parameters = theta_c

        pdis_d_arr,pdis_d = PDIS_D(self._train_c, self._policy_c, self._gamma)
        sigma_s = np.std(pdis_d_arr)
        safety_term = (sigma_s/(np.sqrt(self._train_s_size)))*stats.t.ppf(1-self._delta,self._train_s_size-1)
        
        print("mean:",pdis_d)
        print("variance:",sigma_s)
        
        safetyVal = pdis_d - safety_term
        
        if(safetyVal>=self._c):
            return True, pdis_d
        else:
            return False, pdis_d