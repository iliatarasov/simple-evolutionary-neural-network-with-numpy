import random
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from PIL import Image
import numpy as np

from .entity import Agent

CMAP = colors.ListedColormap(['darkblue', 'powderblue', 'violet'])

def play_round(agent: Agent, 
               field_size: int=10, 
               fail_reward: float=0.5, 
               time_limit: int=None) -> float:
    '''
    Simulates one round of gameplay
    Arguments:
        agent: playing agent
        field_size (int): size of field
        fail_reward (float): percentage of reward in case of failure
        time_limit (int): time limit per round
    '''
    
    if time_limit is None:
        time_limit = int(field_size * 2.5)
        
    food_exists = False
    time_remaining = time_limit
    agent.reset()
    
    while time_remaining > 0:
        
        if not food_exists:
            food_pos = [random.randint(0, field_size - 1), 
            random.randint(0, field_size - 1)]
            while food_pos == agent.pos:
                food_pos = [random.randint(0, field_size - 1), 
                random.randint(0, field_size - 1)]
            food_exists = True
            
        board = np.zeros((field_size, field_size))
        board[tuple(food_pos)] = 1
        board[tuple(agent.pos)] = -1
        action = agent.forward(board)
        
        match action:
            case 0:
                agent.pos[0] = (agent.pos[0] - 1) % field_size
            case 1:
                agent.pos[0] = (agent.pos[0] + 1) % field_size
            case 2:
                agent.pos[1] = (agent.pos[1] - 1) % field_size
            case 3:
                agent.pos[1] = (agent.pos[1] + 1) % field_size
                
        if agent.pos == food_pos:
            food_exists = False
            agent.score += 1
            
        time_remaining -= 1 
        
    if not agent.score:
        agent.score +=  fail_reward * (min_distance(food_pos, 
                                                    agent.pos, 
                                                    field_size)) / field_size
    
    return agent.score


def min_distance(food_pos: list|tuple, 
                 agent_pos: list|tuple, 
                 field_size: int) -> int:
    '''
    Calculates minimal distance in steps towards a food
    on a wraparound square field
    Arguments:
        food_pos (list|tuple): food position
        agent_pos (list|tuple): agent position
        field_size (int): field size
    '''
    a, b = food_pos
    m = field_size
    extended_points = [
        (a - m, b + m), (a, b + m), (a + m, b + m),
        (a - m, b), (a, b), (a + m, b),
        (a - m, b - m), (a, b - m), (a + m, b - m),
    ]
    return min([abs(point[0] - agent_pos[0]) + abs(point[1] - agent_pos[1]) 
             for point in extended_points])
    
    
def vizualize_round(agent: Agent, 
               field_size: int=10, 
               fail_reward: float=0.5, 
               time_limit: int=None,
               path: str=None) -> float:
    '''Same functionality as play_round but saves steps in .png and .gif'''
    matplotlib.use('Agg')
    if time_limit is None:
        time_limit = int(field_size * 2.5)
        
    food_exists = False
    time_remaining = time_limit
    agent.reset()
    
    agent.test_score = 0.0
    
    while time_remaining > 0:
        
        if not food_exists:
            food_pos = [random.randint(0, field_size - 1), 
            random.randint(0, field_size - 1)]
            while food_pos == agent.pos:
                food_pos = [random.randint(0, field_size - 1), 
                random.randint(0, field_size - 1)]
            food_exists = True
            
        board = np.zeros((field_size, field_size))
        board[tuple(food_pos)] = 1
        board[tuple(agent.pos)] = -1
    
        plt.pcolormesh(board, edgecolors='black', cmap=CMAP)
        plt.title(f'Time remaining: {time_remaining} \
            \n score: {agent.test_score}')
        plt.axis('off')
        ax=plt.gca()
        ax.set_aspect('equal')
        plt.savefig(f'{path}step {time_limit - time_remaining}.png')
        plt.close()
        
        action = agent.forward(board)
        
        match action:
            case 0:
                agent.pos[0] = (agent.pos[0] - 1) % field_size
            case 1:
                agent.pos[0] = (agent.pos[0] + 1) % field_size
            case 2:
                agent.pos[1] = (agent.pos[1] - 1) % field_size
            case 3:
                agent.pos[1] = (agent.pos[1] + 1) % field_size
                
        if agent.pos == food_pos:
            food_exists = False
            agent.test_score += 1

        time_remaining -= 1 
    
    images_path = sorted(os.listdir(path), key=lambda x: int(x.split()[1].split('.')[0]))
    images = [Image.open(path+file) for file in images_path]
    
    images[0].save(path+'animation.gif', 
               save_all=True,
               append_images=images[1:],
               duration=250,
               loop=1)
    
    return agent.test_score
