import random
import numpy as np

from .entity import Agent


def play_round(agent: Agent, 
               field_size: int=10, 
               fail_reward: float=0.5, 
               time_limit: float=None):
    
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


def min_distance(food_pos, agent_pos, field_size):
    a, b = food_pos
    m = field_size
    extended_points = [
        (a - m, b + m), (a, b + m), (a + m, b + m),
        (a - m, b), (a, b), (a + m, b),
        (a - m, b - m), (a, b - m), (a + m, b - m),
    ]
    return min([abs(point[0] - agent_pos[0]) + abs(point[1] - agent_pos[1]) 
             for point in extended_points])