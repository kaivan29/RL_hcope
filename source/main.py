import numpy as np
import cma
import time
import csv
from itertools import product

#Import User Defined files
from policy_improvement import PolicyImprovement
from parse_data import parse_data, write_policy_to_file
from softmax import Softmax
from pdis import PDIS_D

np.random.seed(111)

def policyGen():
    #Parse data.csv
    print("\nReading data.csv")
    numStates, numActions, numEpisodes, D = parse_data("data.csv")
    theta_b = np.random.rand(numStates, numActions)
    #numEpisodes = len(D)
    print("Data reading done of episodes: ", numEpisodes)

    policy_e = Softmax(numStates,numActions)
    delta = 0.05
    c = 1.41537

    gamma = 0.95
    sigma = 0.5
    n = 100
    i = 0
    iterations = 0
    print("Start to generate policies\n\n")
    while i < n:
        start_time = time.time()
        iterations += 1
        print("Iteration: ",iterations)
        # split from randomly sampled data
        D_sample = np.random.choice(D,numEpisodes,replace=False)
        # 100k for training
        train_c = D_sample[:int(numEpisodes*0.1)]
        # 60k for testing
        test_s = D_sample[int(numEpisodes*0.94):]

        # generate policies
        candidate_policy = PolicyImprovement(D, train_c, test_s, theta_b, policy_e, delta, c, gamma)
        # evaluate
        selected_theta, result = candidate_policy.evaluate(sigma)
        print("Result: ", result)
        
        #conduct a safety test
        safety_pass=False
        if result < 100:
            print("\n\nTheta found and performing safety test")
            policy_c = Softmax(numStates, numActions)
            policy_c.parameters = selected_theta
            safety_pass, safety_pdis = candidate_policy.safetyTest(theta_c = selected_theta, policy_c = policy_c)
            print("safety pass:", safety_pass)
        
        #output results and selected theta
        print(f'Safety pass checked and theta returned in time: {time.time() - start_time}')
        if safety_pass == True:
            # print("Theta(4s+a): \n", selected_theta)
            print("Writing returns and policy to file")
            #write to output
            write_policy_to_file("../policy" + str(i + 1) + ".txt", selected_theta)
            i += 1
        else:
            print("Safety pass failed")
            print("returns: ", result)
            
        print("Number of theta obtained so far: ", i)
        print("\n")

    print("Total number of theta obtained: ", i)
    
def main():
    policyGen()

if __name__ == "__main__":
    main()
