import numpy as np
import cma
import time
import csv
from itertools import product

#Import User Defined files
from candidate_selection import CandidateSelection
from parse_data import parse_data, output_returns_to_file, write_policy_to_file
from tabular_softmax import TabularSoftmax
from pdis import PDIS_D

np.random.seed(333)

def project():
    #Parse data.csv
    numStates,numActions,n,D = parse_data("data.csv")
    theta_b = np.random.rand(numStates, numActions)
    numEpisodes = len(D)
    print("Data reading done of episodes: ", numEpisodes)

    policy_e = TabularSoftmax(numStates,numActions)
    delta = 0.05
    cRatio = 1.41537

    gamma = 0.95
    maxIter = 10
    sigma = 0.5
    results = []
    i = 0
    while i < 100:
        start_time = time.time()
        D_sample = np.random.choice(D,numEpisodes,replace=False)
        train_c = D_sample[:int(numEpisodes*0.8)]
        test_s = D_sample[int(numEpisodes*0.8)+1:]

        candidate_selection = CandidateSelection(D, train_c, test_s, theta_b, policy_e, delta, cRatio, gamma, maxIter)
        
        candidate_theta, result = candidate_selection.evaluateCMAES(sigma)
        
        #conduct a safety test
        safety_pass=False
        if result < 100:
            print("Candidate theta found")
            print("Conducting Safety Test")

            policy_c = TabularSoftmax(numStates, numActions)
            policy_c.parameters = candidate_theta

            safety_pass, safety_pdis = candidate_selection.safetyTest(theta_c = candidate_theta, policy_c = policy_c)
            print("safety_pass:", safety_pass)
            print("Safety_pdis: " + str(safety_pdis))
        
        #output results and candidate theta
        print("return: " + str(result) + " theta: " + str(candidate_theta))
        print("safety_test_pass: " + str(safety_pass))
        print(f'Safety pass checked and candidate returned in time: {time.time() - start_time}')
        if safety_pass == True:
            results.append([result, candidate_theta])
            print("Candidate theta: \n", candidate_theta)
            print("Result: ", result)
            print("Writing returns and policy to file")
            #write to output
            write_policy_to_file("output/policy" + str(i + 1) + ".txt", candidate_theta)
            output_returns_to_file("output/returns_" + str(i + 1) + ".csv", [result])
            #write outside output to save from being overwritten
            write_policy_to_file("../policy" + str(i + 1) + ".txt", candidate_theta)
            output_returns_to_file("../returns_" + str(i + 1) + ".csv", [result])
            print('Done')
            i += 1
        print("Number of theta obtained: " + str(i))

    print("Total number of theta obtained: " + str(i))
        
    print(results)
    
def main():
    project()


if __name__ == "__main__":
    main()
