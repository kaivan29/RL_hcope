import numpy as np
from policy_template import Policy
from typing import Union

class TabularSoftmax(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self.numActions = numActions
        self.numStates = numStates
        self._theta = np.zeros((numStates, numActions))
        self._p = None
        
        #TODO
        # pass

    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)
        
    def get_probabilities(self, s_list, a_list):
        """
        Given a state-action pair list, returns an 1D numpy array containing the 
        probabilities corresponding to those pairs.
        """
        list_of_vals = []
        for i in range(len(s_list)):
            list_of_vals.append(self.getActionProbabilities(s_list[i])[a_list[i]])

        return np.array(list_of_vals)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        
        #TODO
        print("inside call")
        action_prob = self.getActionProbabilities(state)
        
        if action is None:
            return action_prob
        else:
            return action_prob[action]
        
        # pass

    def samplAction(self, state:int)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        action = np.random.choice (self.numActions, p = self.getActionProbabilities(state))
        
        return action
        #TODO
        # pass
        
#     def pi(self, state:np.ndarray, action)->np.ndarray:
#         #print("inside pi")
#         action_prob = self.getActionProbabilities(state)
#         return action_prob[action]

    def getActionProbabilities(self, state:int)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        #print("in get action P")
        #state = state.astype(int)
        #TODO
        if self._theta is not None:
            x = self._theta[int(state)][:] # np.dot(self._theta, self._phi)
            self._p = x - np.max(x)
            
            exp_state_row = np.exp(self._p)
            
            probabilities = exp_state_row / np.sum(exp_state_row)
        else: # for pi_b
            probabilities = self._p[int(state)][:]
            # print(len(probabilities))

        return probabilities

        #TODO
        # pass
