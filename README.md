This is a learning project that aims to build an elementary evolutionary neural network to play a simple game of survival. The main idea behind this work is to create a trainable neural network without backpropagation while relying only on random mutations.

The game consists of an environment in the form of a wraparound square field and two types of cells: agents and food. The objective is simple: an agent must collect as many food cells as it can within the given time. 

![game_state](https://raw.githubusercontent.com/iliatarasov/simple-evolutionary-neural-network-with-numpy/main/data/agents/step%200.png)

Neural network agents collect food, best ones get to reproduce and pass their 'genes' to theis successors. Each new generation receives random mutations and the hope is that this process will direct the network towards a well-trained state.

IMPORTANT: Python version >= 3.11 is required.