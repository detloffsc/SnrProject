import tensorflow as tf
import numpy as np

def generate_maze(size):
    maze = np.random.choice([0, 1], size=size)
    maze[0, 0] = 0  # Start
    maze[size[0] - 1, size[1] - 1] = 0  # End
    return maze

def create_training_data(num_mazes, size):
    inputs = []
    targets = []
    for _ in range(num_mazes):
        maze = generate_maze(size)
        inputs.append(maze.flatten())
        # For simplicity, we're not using targets in this example
        targets.append(1)
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
model.fit(X_train, X_train, epochs=10)

# Function to solve the maze using the trained model
def solve_maze(model, maze):
    input_data = maze.flatten().reshape(1, -1)
    prediction = model.predict(input_data)
    solved_maze = prediction.reshape(maze.shape)
    return solved_maze

# Example usage
maze = generate_maze(size)
print("Original Maze:")
print(maze)

solved_maze = solve_maze(model, maze)
print("Solved Maze:")
print(solved_maze)
