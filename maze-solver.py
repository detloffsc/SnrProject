import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Function to generate a solvable maze using backtracking algorithm
def generate_solvable_maze(size):
    maze = np.ones(size, dtype=np.int8)
    start = (0, 0)
    end = (size[0] - 1, size[1] - 1)
    stack = [start]
    maze[start] = 0

    while stack:
        current = stack[-1]
        if current == end:
            break
        neighbors = []
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= next_cell[0] < size[0] and 0 <= next_cell[1] < size[1] and maze[next_cell] == 1:
                neighbors.append(next_cell)

        if neighbors:
            next_cell = neighbors[np.random.randint(0, len(neighbors))]
            stack.append(next_cell)
            maze[next_cell] = 0
        else:
            stack.pop()

    return maze

# Function to create training data
def create_training_data(num_mazes, size):
    inputs = []
    targets = []
    for _ in range(num_mazes):
        maze = generate_solvable_maze(size)
        inputs.append(maze.flatten())
        targets.append(maze.flatten())  # Using the maze itself as the target for simplicity
    return np.array(inputs), np.array(targets)

# Generate training data
num_mazes = 1000
size = (10, 10)
X_train, y_train = create_training_data(num_mazes, size)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(size[0] * size[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(size[0] * size[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=30)

# Function to solve the maze using the trained model
def solve_maze(model, maze):
    input_data = maze.flatten().reshape(1, -1)
    prediction = model.predict(input_data)
    solved_maze = (prediction > 0.5).astype(int).reshape(maze.shape)
    return solved_maze

# Function to visualize mazes
def plot_mazes(original_maze, solved_maze):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_maze, cmap='gray')
    axs[0].set_title('Original Maze')
    axs[1].imshow(solved_maze, cmap='gray')
    axs[1].set_title('Solved Maze')
    plt.show()

# Example usage
maze = generate_solvable_maze(size)
print("Original Maze:")
print(maze)

solved_maze = solve_maze(model, maze)
print("Solved Maze:")
print(solved_maze)

plot_mazes(maze, solved_maze)
