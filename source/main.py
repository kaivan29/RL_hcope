import numpy as np
import cma
import time
import csv
from itertools import product

#Import User Defined files
from policy_improvement import PolicyImprovement
from parse_data import parse_data, write_policy_to_file
from softmax import Softmax

np.random.seed(111)

def policyGen():
    #Parse data.csv
    print("\nReading data.csv")
    numStates, numActions, n, Data = parse_data("data.csv")
    n = len(Data)
    theta_b = np.random.rand(numStates, numActions)
    print("Data reading done of episodes: ", n)

    policy_e = Softmax(numStates,numActions)
    delta = 0.05
    c = 1.41537

    gamma = 0.95
    sigma = 0.5
    runs = 100
    i = 0
    iterations = 0
    print("Start to generate policies\n\n")
    while i < runs:
        start_time = time.time()
        iterations += 1
        print("Iteration: ",iterations)
        # split from randomly sampled data
        Data_sample = np.random.choice(Data, n,replace=False)
        # 100k for training
        train_limit = int(n*0.1)
        #print(train_limit)
        train_c = Data_sample[:train_limit]
        # 60k for testing
        test_limit = int(n*0.94)
        test_s = Data_sample[test_limit:]

        # generate policies
        candidate_policy = PolicyImprovement(Data, train_c, test_s, theta_b, policy_e, delta, c, gamma)
        # evaluate
        selected_theta, result = candidate_policy.evaluate(sigma)
        print("Result: ", result)
        
        #conduct a safety test
        safety_pass=False
        if result < 100:
            print("\nTheta found and performing safety test")
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
