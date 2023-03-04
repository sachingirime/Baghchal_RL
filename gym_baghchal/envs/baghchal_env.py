import gym
from gym import spaces
import numpy as np

class BaghchalEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ascii']}
    
    def __init__(self):
        self.board_size = 5
        self.num_tigers = 4
        self.num_goats = 20
        
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.tigers_positions = []
        self.goats_positions = []
        self.turn = 1  # 1: tigers, -1: goats
        self.done = False
        
        self.action_space = spaces.Tuple((spaces.Discrete(self.board_size), spaces.Discrete(self.board_size)))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=int)
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.tigers_positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
        self.goats_positions = [(i, j) for i in range(1, self.board_size-1) for j in range(1, self.board_size-1) if (i+j) % 2 == 0]
        for i, j in self.tigers_positions:
            self.board[i, j] = 1
        for i, j in self.goats_positions:
            self.board[i, j] = -1
        self.turn = 1
        self.done = False
        return self.board.copy()
    
    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, {}
        i, j = action
        reward = 0
        if self.turn == 1:  # tigers turn
            if (i, j) in self.tigers_positions:
                return self.board.copy(), 0, False, {}
            if self.board[i, j] != 0:
                return self.board.copy(), 0, False, {}
            valid_moves = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            valid_moves = [(p, q) for p, q in valid_moves if 0 <= p < self.board_size and 0 <= q < self.board_size]
            for p, q in valid_moves:
                if self.board[p, q] == -1:
                    middle_i, middle_j = (i+p)//2, (j+q)//2
                    if (middle_i, middle_j) in self.tigers_positions:
                        self.board[i, j] = 1
                        self.board[middle_i, middle_j] = 0
                        self.tigers_positions.remove((middle_i, middle_j))
                        self.tigers_positions.append((i, j))
                        reward = 1
                        break
            else:  # no goat was eaten
                self.board[i, j] = 1
                for p, q in self.tigers_positions:
                    if (p, q) != (i, j):
                        self.board[p, q] = 0
                self.tigers_positions = [(i, j)]
        else:  # goats turn
            if (i, j) in self.goats_positions:
                return self.board.copy(), 0, False, {}
            if self.board[i, j] != 0:
                return self.board.copy(), 0, False, {}
            self.board[i, j] = -1
            self.goats_positions.remove((i, j))
            if len(self.goats_positions) == 0:
                self.done = True
                reward = 10
            self.turn = -1
        return self.board.copy(), reward, self.done, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(self.board)
        elif mode == 'ascii':
            print('  ' + ' '.join(str(i) for i in range(self.board_size)))
            for i in range(self.board_size):
                row = ''.join('.' if self.board[i, j] == 0 else 'T' if (i, j) in self.tigers_positions else 'G' for j in range(self.board_size))
                print(i, row)
        else:
            super(BaghchalEnv, self).render(mode=mode)

    def close(self):
        pass





#The `__init__` method initializes the environment, with the board size, number of tigers, and number of goats. The board is represented as a numpy array of integers, where 0 means empty, 1 means tiger, and -1 means goat. The `turn` variable keeps track of whose turn it is (tigers or goats), and the `done` variable is set to `True` when the game is over.

#The `reset` method resets the environment to its initial state, with the tigers and goats positioned in their starting positions.

#The `step` method receives an action (a tuple of integers representing a row and a column), and updates the board state accordingly. If the action is invalid (e.g. trying to move to an occupied cell), no change is made to the board. If it is a valid move, and the tigers can eat a goat, they do so and receive a reward of 1. If a goat reaches the other side of the board, the game is over and the goats win with a reward of 10.

#The `render` method can display the board in either "human" or "ascii" mode. In "human" mode, it prints the numpy array to the console. In "ascii" mode, it prints a more visually appealing representation of the board.

#The `close` method is empty, as it is not needed for this environment.

