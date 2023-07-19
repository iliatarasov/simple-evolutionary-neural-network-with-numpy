import pickle
import random
import numpy as np

from .play import play_round
from .entity import Agent

class Evo:
    def __init__(self,
                 n_agents: int=100,
                 field_size: int=10) -> None:
        self.agents = [Agent(field_size=field_size, primarch=True) 
                       for _ in range(n_agents)]
        self.n_agents = n_agents
        self.field_size = field_size
    
    def train(self,
              n_generations: int=20000,
              rounds_per_agent: int=10,
              mutation_type: str='adaptive', #'adaptive' or 'constant'
              initial_mutation_rate: float=0.3,
              pairing_type: str='best', # 'best' or 'random'
              reproducing_percentage: float=0.25,
              reproduction_type: str='asexual', #'sexual' or 'asexual'
              output_path: str='results.pkl') -> None:
        best_agents = []
        high_scores = []
        mean_scores = []

        if mutation_type == 'constant':
            mutation_rate = initial_mutation_rate

        for generation in range(1, n_generations + 1):
            if mutation_type == 'adaptive':
                mutation_rate = (n_generations - 
                                generation) / n_generations * initial_mutation_rate
            
            #food gathering
            scores = []
            for agent in self.agents:
                scores.append(np.mean([play_round(agent, 
                                                  field_size=self.field_size) 
                                    for _ in range(rounds_per_agent)]))
            mean_scores.append(np.mean(scores))

            #sorting by performance
            n_reproducing = int(self.n_agents * reproducing_percentage)
            reproducing_agents = sorted(zip(self.agents, scores), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:n_reproducing]
            if pairing_type == 'random':
                random.shuffle(reproducing_agents)
            #tracking the best in generation
            best_agents.append(reproducing_agents[0][0])
            high_scores.append(reproducing_agents[0][1])

            #reproduction and mutation
            self.agents = []
            parent = iter(reproducing_agents)
            for parent1, parent2 in zip(parent, parent):
                for _ in range(int(1 / reproducing_percentage)):
                    child = parent1[0].reproduce(parent2[0], 
                                                 type=reproduction_type)
                    #child = parent1[0].reproduce(type='asexual')
                    child.mutate(mutation_rate=mutation_rate)
                    self.agents.append(child)
            if generation % 1000 == 0:
                print(f'Generation {generation}/{n_generations}\t', 
                      f'high score: {max(high_scores)}')

        output = {
            'best agents': best_agents,
            'high scores': high_scores,
            'mean scores': mean_scores,
        }

        if not output_path.endswith('.pkl'):
            output_path += '.pkl'
            
        with open(output_path, 'wb') as file:
            pickle.dump(output, file)