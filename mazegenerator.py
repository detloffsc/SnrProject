import json
import numpy as np

maze = {
    "maze_id" : 1,
    "maze" : [
            [ 1,  0,  1,  1,  1,  1,  1,  1],
            [ 1,  0,  1,  1,  1,  0,  1,  1],
            [ 1,  1,  1,  1,  0,  1,  0,  1],
            [ 1,  1,  1,  0,  1,  1,  1,  1],
            [ 1,  1,  0,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  0,  1,  0,  0,  0],
            [ 1,  1,  1,  0,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  0,  1,  1,  1]
        ]
}

rand_maze = np.random.randint(2, size=(8, 8))

print(rand_maze)
#json_object = json.dumps(rand_maze, indent=4)

#with open("test.json", "w") as outfile:
#    outfile.write(json_object)