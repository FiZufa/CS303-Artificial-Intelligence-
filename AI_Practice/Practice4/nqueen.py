import numpy as np
import time
import random

N = 100 # You shoule change the test size here !!!

def my_range(start, end):
    if start <= end:
        return range(start, end + 1)
    else:
        return range(start, end - 1, -1)


class Problem:
    char_mapping = ('·', 'Q')

    def __init__(self, n=4):
        self.n = n

    def is_valid(self, state):
        """
        check the state satisfy condition or not.
        :param state: list of points (in (row id, col id) tuple form)
        :return: bool value of valid or not
        """
        board = self.get_board(state)
        res = True
        for point in state:
            i, j = point
            condition1 = board[:, j].sum() <= 1
            condition2 = board[i, :].sum() <= 1
            condition3 = self.pos_slant_condition(board, point)
            condition4 = self.neg_slant_condition(board, point)
            res = res and condition1 and condition2 and condition3 and condition4
            if not res:
                break
        return res

    def is_satisfy(self, state):
        """
        check if the current board meets the win condition
        """
        return self.is_valid(state) and len(state) == self.n

    def pos_slant_condition(self, board, point):
        i, j = point
        tmp = min(self.n - i - 1, j)
        start = (i + tmp, j - tmp)
        tmp = min(i, self.n - j - 1)
        end = (i - tmp,  j + tmp)
        rows = my_range(start[0], end[0])
        cols = my_range(start[1], end[1])
        return board[rows, cols].sum() <= 1

    def neg_slant_condition(self, board, point):
        i, j = point
        tmp = min(i, j)
        start = (i - tmp, j - tmp)
        tmp = min(self.n - i - 1, self.n - j - 1)
        end = (i + tmp, j + tmp)
        rows = my_range(start[0], end[0])
        cols = my_range(start[1], end[1])
        return board[rows, cols].sum() <= 1

    def get_board(self, state):
        board = np.zeros([self.n, self.n], dtype=int)
        for point in state:
            board[point] = 1
        return board

    def print_state(self, state):
        board = self.get_board(state)
        print('_' * (2 * self.n + 1))
        for row in board:
            for item in row:
                print(f'|{Problem.char_mapping[item]}', end='')
            print('|')
        print('-' * (2 * self.n + 1))

    def count_conflicts(self, position, state):
        row, col = position
        conflicts = 0

        # Check horizontal conflicts (same row)
        for j in range(self.n):
            if j != col and state[j][0] == row:
                conflicts += 1

        # Check vertical conflicts (same column)
        for j in range(self.n):
            if j != col and state[j][1] == state[col][1]:
                conflicts += 1

        # Check diagonal conflicts (both diagonals)
        for j in range(self.n):
            if j != col:
                if abs(state[j][0] - row) == abs(state[j][1] - state[col][1]):
                    conflicts += 1

        return conflicts

# def bts(problem):
#     action_stack = []
#     while not problem.is_satisfy(action_stack):
#         # TODO: Implement backtracking search logic here
#         yield action_stack

def bts(problem):
    action_stack = []
    n = problem.n

    def backtrack(row):
        if row == n:
            yield list(action_stack)
        else:
            for col in range(n):
                if problem.is_valid(action_stack + [(row, col)]):
                    action_stack.append((row, col))
                    yield from backtrack(row + 1)
                    action_stack.pop()

    yield from backtrack(0)
    


def min_conflict(problem, max_steps=10**10):
    action_stack = initialize_state(problem)
    print(action_stack)
    n = problem.n

    for _ in range(max_steps):
        if problem.is_satisfy(action_stack):
            yield action_stack
            return

        # Find a queen with conflicts
        conflicted_queen = find_conflicted_queen(problem, action_stack)
        # print('Conflict = ', problem.count_conflicts)

        if conflicted_queen is None:
            yield action_stack
            return

        # Move the conflicted queen to a position with fewer conflicts
        new_position = minimize_conflicts(problem, action_stack, conflicted_queen)
        # print(new_position)
        action_stack[conflicted_queen] = new_position

    yield action_stack

def initialize_state(problem):
    state = []
    for i in range(problem.n):
        row = random.randint(0, problem.n-1) 
        state.append((row, i))
        
    return state


def find_conflicted_queen(problem, state):
    conflicts = [problem.count_conflicts((state[i][0], state[i][1]), state) for i in range(problem.n)]
    
    conflicted_queens = [i for i in range(problem.n) if conflicts[i] > 0]
    
    if conflicted_queens:
        return random.choice(conflicted_queens)
    return None


def minimize_conflicts(problem, state, conflicted_queen):
    min_position = state[conflicted_queen]
    min_conflicts = problem.count_conflicts(min_position, state)

    for position in range(problem.n):
        if position == state[conflicted_queen]:
            continue
        conflicts = problem.count_conflicts((position, state[conflicted_queen][1]), state)
        if conflicts < min_conflicts:
            min_position = (position, state[conflicted_queen][1])
            min_conflicts = conflicts
        elif conflicts == min_conflicts:
            if random.random() < 0.5:
                min_position = (position, state[conflicted_queen][1])
    
    state[conflicted_queen] = min_position 

    return min_position


# test_block

# test_block
n = N # Do not modify this parameter, if you want to change the size, go to the first line of whole program.
render = False # here to select GUI or not
method = min_conflict  # here to change your method; bts or improving_bts or min_conflict
p = Problem(n)
if render:
    import pygame
    w, h = 90 * n + 10, 90 * n + 10
    screen = pygame.display.set_mode((w, h))
    screen.fill('white')
    action_generator = method(p)
    clk = pygame.time.Clock()
    queen_img = pygame.image.load('./queen.png')
    while True:
        for event in pygame.event.get():
            if event == pygame.QUIT:
                exit()
        try:
            actions = next(action_generator)
            screen.fill('white')
            for i in range(n + 1):
                pygame.draw.rect(screen, 'black', (i * 90, 0, 10, h))
                pygame.draw.rect(screen, 'black', (0, i * 90, w, 10))
            for action in actions:
                i, j = action
                screen.blit(queen_img, (10 + 90 * j, 10 + 90 * i))
            pygame.display.flip()
        except StopIteration:
            pass
        clk.tick(5)
    pass
else:
    start_time = time.time()
    for actions in method(p):
        pass
    p.print_state(actions)
    print(time.time() - start_time, 'seconds')

