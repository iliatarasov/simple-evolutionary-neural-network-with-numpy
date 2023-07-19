from typing import Self
import numpy as np

class Agent:
    def __init__(self, field_size: int=20, primarch: bool=False) -> None:
        if primarch:
            self.weights = np.random.randn(field_size ** 2, 4)
            self.biases = np.zeros(4)
        self.score = 0
        self.field_size = field_size
        
    def forward(self, board: np.ndarray) -> int:
        value = np.matmul(board.flatten().T, self.weights) + self.biases
        softmax = np.exp(value) / (np.exp(value).sum() + 1e-6)
        return np.argmax(softmax)
        
    def mutate(self, mutation_rate: float=0.01) -> None:
        self.weights += np.random.randn(*self.weights.shape) * mutation_rate
        self.biases += np.random.randn(*self.biases.shape) * mutation_rate
        
    def reset(self) -> None:
        middle = self.field_size // 2
        self.pos = [middle, middle]
        
    def reproduce(self, parent2: Self=None, primary_bias: float=0.6, 
                  *, type: str='asexual') -> Self:
        if type == 'asexual':
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
        