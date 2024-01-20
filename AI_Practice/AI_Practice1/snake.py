import numpy as np

def snake(map_data, actions):
    rows = len(map_data)
    cols = len(map_data[0])
    snake_head = None
    snake_body = []

    for r in range(rows):
        for c in range(cols):
            if map_data[r][c] == '@':
                snake_head = (r, c)
                snake_body.append(snake_head)
            if map_data[r][c] == '#':
                snake_body.append((r,c))

    t=0
    for a in actions:
        t+=1
        if a == 0:  # Move up
            new_head = (snake_head[0]-1, snake_head[1])
        elif a == 1:  # Move down
            new_head = (snake_head[0]+1, snake_head[1])
        elif a == 2:  # Move left
            new_head = (snake_head[0], snake_head[1]-1)
        elif a == 3:  # Move right
            new_head = (snake_head[0], snake_head[1]+1)

        
        if(new_head[0] < 0 or new_head[0] >= rows 
           or new_head[1] < 0 or new_head[1] >= cols 
           or map_data[new_head[0]][new_head[1]] == 'x'
           or map_data[new_head[0]][new_head[1]] == "#"
           ):
            return t
        
        # the head become body
        snake_body.append(snake_head)
        map_data[new_head[0]][new_head[1]] = '@'
        map_data[snake_head[0]][snake_head[1]] = '#'

        #remove the tail
        if len(snake_body) > 1:
            tail = snake_body.pop(0)
            map_data[tail[0]][tail[1]] = '-'

        snake_head = new_head

    return f"{snake_head[0]} {snake_head[1]}"

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