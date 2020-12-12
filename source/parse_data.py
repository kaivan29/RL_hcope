import numpy as np
import csv
import itertools

def write_policy_to_file(file_path, data):
    with open(file_path, 'a') as file:
        file.write('\n'.join([str(i) for i in data]))
        
def parse_data(file_name):
    Data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        num_states = 18
        num_actions = 4
        counter = 0
        line_count = 0
        next_epi = 0
        for row in csv_reader:
            #total number of episodes
            if(next_epi==0):
                episodes = row
                next_epi = 1

            #Number of timesteps for first episode
            elif(next_epi==1):
                # initialize everything
                time = row
                next_epi = 1 + int(time[0])
                line_count = 1
                counter = 1
                H = {}
                S = []; A = []; R = []; PI = []

            # At each episode
            elif line_count == next_epi:
                if counter % 100000 == 0:
                    print("Episodes done: ", counter)
                # update
                H['S']=np.array(S)
                H['A']=np.array(A)
                H['R']=np.array(R)
                H['PI']=np.array(PI)
                Data.append(H)
                # re-initialize everything
                time = row
                next_epi = 1 + int(time[0])
                line_count = 1
                H ={}
                S = []; A = []; R = []; PI = []
                counter += 1
            
            # read all the data from time = 0 to tim = T for this episode
            elif line_count < next_epi:
                line_count += 1
                S.append(float(row[0]))
                A.append(int(row[1]))
                R.append(float(row[2]))
                PI.append(float(row[3]))

    return num_states, num_actions, counter, Data

def main():
    parse_data("data.csv")

if __name__ == '__main__':
    main()
