import numpy as np
import cma
import time
import csv
from itertools import product

#Import User Defined files
from candidate_selection import CandidateSelection
from parse_data import parse_data, output_returns_to_file, write_policy_to_file
from softmax import Softmax
from pdis import PDIS_D

np.random.seed(111)

def project():
    #Parse data.csv
    numStates,numActions,n,D = parse_data("data.csv")
    theta_b = np.random.rand(numStates, numActions)
    numEpisodes = len(D)
    print("Data reading done of episodes: ", numEpisodes)

    policy_e = Softmax(numStates,numActions)
    delta = 0.05
    c = 1.41537

    gamma = 0.95
    sigma = 0.5
    i = 0
    while i < 100:
        start_time = time.time()
        print("Iteration: ",i+1)
        # split from randomly sampled data
        D_sample = np.random.choice(D,numEpisodes,replace=False)
        # 100k for training
        train_c = D_sample[:int(numEpisodes*0.1)]
        # 60k for testing
        test_s = D_sample[int(numEpisodes*0.94):]

        candidate_selection = CandidateSelection(D, train_c, test_s, theta_b, policy_e, delta, c, gamma)
        candidate_theta, result = candidate_selection.evaluateCMAES(sigma)
        print("Result: ", result)
        
        #conduct a safety test
        safety_pass=False
        if result < 100:
            print("\n\nTheta found and performing safety test")
            policy_c = Softmax(numStates, numActions)
            policy_c.parameters = candidate_theta
            safety_pass, safety_pdis = candidate_selection.safetyTest(theta_c = candidate_theta, policy_c = policy_c)
            print("safety pass:", safety_pass)
        
        #output results and candidate theta
        print(f'Safety pass checked and candidate returned in time: {time.time() - start_time}')
        if safety_pass == True:
            # print("Theta(4s+a): \n", candidate_theta)
            print("Writing returns and policy to file")
            #write to output
            write_policy_to_file("../policy" + str(i + 1) + ".txt", candidate_theta)
            #output_returns_to_file("../returns_" + str(i + 1) + ".csv", [result])
            print('Done\n\n')
            i += 1
        else:
            print("Safety pass failed")
            print("returns: ", result)
            
        print("Number of theta obtained so far: " + str(i))

    print("Total number of theta obtained: " + str(i))
    
def main():
    project()


if __name__ == "__main__":
    main()
