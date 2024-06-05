from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
import tensorflow as ts
import json

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.python.keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience

plt.show()


#maze = np.array([
#    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
#    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
#    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
#    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
#    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
#    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
#])

with open('maze.json') as mazes:
    maze = json.load(mazes)


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows-1, ncols-1] = 0.9 # treasure cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


# Exploration factor
epsilon = 0.1

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

qmaze = TreasureMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)

def play_game(model, qmaze, pirate_cell):
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False
        
def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True

def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

def qtrain(model, maze, **opt):

    # exploration factor
    global epsilon 
    epsilon = opt.get('epsilon', 1.0)
    # number of epochs
    n_epoch = opt.get('n_epoch', 150)

    # maximum memory to store episodes
    max_memory = opt.get('max_memory', 1000)

    # maximum data size for training
    data_size = opt.get('data_size', 50)

    # start time
    start_time = datetime.datetime.now()

    # Construct environment/game from numpy array: maze (see above)
    qmaze = TreasureMaze(maze)

    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)
    
    win_history = []   # history of win/lose game
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0

    # pseudocode:
    # For each epoch:
    #    Agent_cell = randomly select a free cell
    #    Reset the maze with agent set to above position
    #    Hint: Review the reset method in the TreasureMaze.py class.
    #    envstate = Environment.current_state
    #    Hint: Review the observe method in the TreasureMaze.py class.
    #    While state is not game over:
    #        previous_envstate = envstate
    #        Action = randomly choose action (left, right, up, down) either by exploration or by exploitation
    #        envstate, reward, game_status = qmaze.act(action)
    #    Hint: Review the act method in the TreasureMaze.py class.
    #        episode = [previous_envstate, action, reward, envstate, game_status]
    #        Store episode in Experience replay object
    #    Hint: Review the remember method in the GameExperience.py class.
    #        Train neural network model and evaluate loss
    #    Hint: Call GameExperience.get_data to retrieve training data (input and target) and pass to model.fit method 
    #          to train the model. You can call model.evaluate to determine loss.
    #    If the win rate is above the threshold and your model passes the completion check, that would be your epoch.
    for epoch in range(n_epoch):
        agent_cell = qmaze.free_cells[np.random.randint(0, len(qmaze.free_cells))]
        
        qmaze.reset(agent_cell)
        
        envstate = qmaze.observe()
        
        n_episodes = 0
        loss = 0.0
        
        while True:
            prev_envstate = envstate
            
            if np.random.rand() < epsilon:
                action = np.random.choice([LEFT, UP, RIGHT, DOWN])
            else:
                q = model.predict(prev_envstate)
                action = np.argmax(q[0])
                
            envstate, reward, game_status = qmaze.act(action)
            
            episode = [prev_envstate, action, reward, envstate, game_status]
            
            experience.remember(episode)
            
            inputs, targets = experience.get_data(data_size=data_size)
            if inputs is not None:
                loss += model.train_on_batch(inputs, targets)
                
            n_episodes += 1
            
            if game_status == 'win':
                win_history.append(1)
                win_rate = sum(win_history[-hsize:]) / hsize
                break
            elif game_status == 'lose':
                win_history.append(0)
                win_rate = sum(win_history[-hsize:]) / hsize
                break

    #Print the epoch, loss, episodes, win count, and win rate for each epoch
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # We simply check if training has exhausted all free cells and if in all
        # cases the agent won.
        if win_rate > 0.9 : epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Determine the total time for training
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)
    
qmaze = TreasureMaze(maze)
show(qmaze)

model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)

completion_check(model, qmaze)
show(qmaze)

pirate_start = (0, 0)
play_game(model, qmaze, pirate_start)
show(qmaze)