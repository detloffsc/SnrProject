import json

with open('maze.json') as mazes:
    maze = json.load(mazes)

print(maze)