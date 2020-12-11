import numpy as np
from policy_template import Policy
from typing import Union

class Softmax(Policy):
    """
    A Softmax Policy
    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int):
        self.numActions = numActions
        self.numStates = numStates
        self._theta = np.zeros((numStates, numActions))
        self._p = None

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
        action_prob = self.getActionProbabilities(state)
        
        if action is None:
            return action_prob
        else:
            return action_prob[action]

    def samplAction(self, state:int)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        action = np.random.choice (self.numActions, p = self.getActionProbabilities(state))
        
        return action
        
    def getActionProbabilities(self, state:int)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        if self._theta is None:
            probabilities = self._p[int(state)][:]
        else:
            theta_s = self._theta[int(state)][:]
            self._p = theta_s - np.max(theta_s)
            exp_state = np.exp(self._p)
            probabilities = exp_state / np.sum(exp_state)
        '''
        if self._theta is not None:
            theta_s = self._theta[int(state)][:]
            self._p = theta_s - np.max(theta_s)
            
            exp_state = np.exp(self._p)
            
            probabilities = exp_state / np.sum(exp_state)
        else: # for pi_b
            probabilities = self._p[int(state)][:]
        '''
        return probabilities
