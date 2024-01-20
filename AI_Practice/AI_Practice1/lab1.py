import numpy as np

def snake(map, actions):

    rows, cols = len(map), len(map[0])
    head = None # (row, column)
    snake_body = [] # list position of the snake's body


    for r in range(rows):
        for c in range(cols):
            if map[r][c] == '@':
                head = (r, c)
         
            
    # i = time 
    for i, action in enumerate(actions):

        if action == 0:  # up
            new_head = (head[0] - 1, head[1])
        elif action == 1:  # down
            new_head = (head[0] + 1, head[1])
        elif action == 2:  # left
            new_head = (head[0], head[1] - 1)
        elif action == 3:  # right
            new_head = (head[0], head[1] + 1)

        if (
            new_head[0] < 0 or new_head[0] >= rows 
            or new_head[1] < 0 or new_head[1] >= cols 
            or map[new_head[0]][new_head[1]] == 'x'
            or new_head in snake_body):

            return i

        #print("head: ", head)
  
        snake_body.insert(0, head)
        head = new_head
        #print("body: ", snake_body)
        #for segment in snak

    return f"{head[0]}, {head[1]}"


map_data = np.array([
    list("---------"),
    list("------x--"),
    list("-x-------"),
    list("---@-----"),
    list("---##----"),
    list("------x--"),
    list("--x----x-"),
    list("-x-------"),
    list("---------")
], dtype=str)
actions = np.array([0, 0, 3, 3, 0, 3, 3, 1, 1, 1, 1, 1, 3, 1, 1, 2, 2, 2, 2, 2], dtype=int)

result = snake(map_data, actions)
print(result)