from typing import Self
import numpy as np

class Agent:
    '''Agent class for evolutionary network'''
    def __init__(self, field_size: int=10, primarch: bool=False) -> None:
        '''
        Arguments:
            field_size (int): field size
            primarch (bool): if True, weights are initialized (used for 
                            first generation agent creation)
        '''
        if primarch:
            self.weights = np.random.randn(field_size ** 2, 4)
            self.biases = np.zeros(4)
        self.score = 0
        self.field_size = field_size
        
    def forward(self, board: np.ndarray) -> int:
        '''Forward pass'''
        value = np.matmul(board.flatten().T, self.weights) + self.biases
        softmax = np.exp(value) / (np.exp(value).sum() + 1e-6)
        return np.argmax(softmax)
        
    def mutate(self, mutation_rate: float=0.01) -> None:
        '''Mutation'''
        self.weights += np.random.randn(*self.weights.shape) * mutation_rate
        self.biases += np.random.randn(*self.biases.shape) * mutation_rate
        
    def reset(self) -> None:
        '''Resets the position of the agent to the middle of the board'''
        middle = self.field_size // 2
        self.pos = [middle, middle]
        
    def reproduce(self, parent2: Self=None, primary_bias: float=0.6, 
                  *, reproduction_type: str='asexual') -> Self:
        '''Reproduction
        Arguments:
            parent2 (Agent): second parent in case of sexual reproduction
            primary_bias (float): percentage of genes taken from the first 
                                  parent
            type (str): 'sexual' for reproduction from 2 parents
                        'asexual' for reproduction from 1 parent
        '''
        if reproduction_type == 'asexual':
            child = self
            child.score = 0.0
            return child
        
        assert parent2 is not None 
        weights_mask = np.random.choice([1, 0], p=[primary_bias, 1 - primary_bias], 
                                        size=self.weights.shape)
        biases_mask = np.random.choice([1, 0], p=[primary_bias, 1 - primary_bias], 
                                        size=self.biases.shape)
        child = Agent(field_size=self.field_size)
        child.weights = self.weights * weights_mask + parent2.weights * np.logical_not(weights_mask)
        child.biases = self.biases * biases_mask + parent2.biases * np.logical_not(biases_mask)
        return child
        