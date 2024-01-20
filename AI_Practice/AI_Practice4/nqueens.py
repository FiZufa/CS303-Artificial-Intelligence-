import numpy as np
import time

N =  8# You shoule change the test size here !!!

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


def bts(problem):
    action_stack = []
    while not problem.is_satisfy(action_stack):
        # TODO: Implement backtracking search logic here
        if len(action_stack) == problem.n:
            yield action_stack[:]
            # Backtrack to the previous row to explore other possibilities
            action_stack.pop()
        elif len(action_stack) < problem.n:
            row = len(action_stack)
            for col in range(problem.n):
                action_stack.append((row, col))
                if problem.is_valid(action_stack):
                    break
                action_stack.pop()
            else:
                if action_stack:
                    # Backtrack to the previous row to explore other possibilities
                    action_stack.pop()
                    row -= 1
        if not action_stack:
            break
        yield action_stack
    
def improving_bts(problem):
    action_stack = []
    remaining_vals = [set(range(problem.n)) for _ in range(problem.n)]
    
    def conflict_count(state, row, col):
        """
        Count the number of conflicts for a given point in the state.
        """
        conflicts = 0
        for r, c in state:
            if r == row or c == col or abs(r - row) == abs(c - col):
                conflicts += 1
        return conflicts

    while not problem.is_satisfy(action_stack):
        if len(action_stack) == problem.n:
            yield action_stack[:]
            row, col = action_stack.pop()
            remaining_vals[row].add(col)
        elif len(action_stack) < problem.n:
            row = len(action_stack)
            possible_cols = list(remaining_vals[row])
            conflict_counts = [conflict_count(action_stack, row, col) for col in possible_cols]

            if conflict_counts:  # Ensure the conflict_counts list is not empty
                next_col = possible_cols[conflict_counts.index(min(conflict_counts))]

                action_stack.append((row, next_col))
                if problem.is_valid(action_stack):
                    remaining_vals[row].remove(next_col)
                else:
                    action_stack.pop()
                    remaining_vals[row].remove(next_col)
                    remaining_vals[row] = {col for col in remaining_vals[row] if col != next_col}
                    if action_stack:
                        row, col = action_stack.pop()
                        remaining_vals[row].add(col)
                        row -= 1
            else:  # Handle the case where conflict_counts is empty
                break
    yield action_stack


import random

def min_conflict(problem):
    action_stack = [random.randint(0, problem.n - 1) for _ in range(problem.n)]  # Initialize with random positions
    max_steps = 1000  # Set a maximum number of steps to prevent infinite loops

    while not problem.is_satisfy(action_stack) and max_steps > 0:
        max_steps -= 1

        conflicted_queens = [i for i in range(problem.n) if problem.conflicts(action_stack, i)]

        if conflicted_queens:
            # Randomly select a conflicted queen
            selected_queen = random.choice(conflicted_queens)

            # Find the position with minimum conflicts for the selected queen
            min_conflict_pos = min(range(problem.n), key=lambda x: problem.conflict_count(action_stack, selected_queen, x))

            # Update the position of the selected queen
            action_stack[selected_queen] = min_conflict_pos

    yield action_stack



# test_block
n = N # Do not modify this parameter, if you want to change the size, go to the first line of whole program.
render = False # here to select GUI or not
method = improving_bts  # here to change your method; bts or improving_bts or min_conflict
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
    print(time.time() - start_time)
