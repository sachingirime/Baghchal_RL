from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
from gym import spaces

class BaghchalEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ascii','rgb_array']}
    def init(self):
        self.board_size = 5
        self.num_tigers = 4
        self.num_goats = 20
        self.action_space = spaces.Tuple((spaces.Discrete(5), spaces.Discrete(5))) # board position
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,5), dtype=int) # 5x5 board with values -1 for goat, 0 for empty and 1 for tiger
        self.board = np.zeros((5,5), dtype=int) # initial board state
        self.board[0,0] = self.board[0,4] = self.board[4,0] = self.board[4,4] = 1 # place tigers in corners
        self.phase = "Placement" # initial game phase is Placement
        self.turn = -1 # initial turn is goat's turn
        self.goats_killed = 0 # initial number of goats killed is 0
        self.tigers_captured = 0 # initial number of tigers captured is 0
        self.goat_positions = [] # initial goat positions is empty
        self.tiger_positions = [(0,0), (0,4), (4,0), (4,4)] # initial tiger positions are at corners
        self.empty_positions = [(i,j) for i in range(5) for j in range(5) if self.board[i,j] == 0] # initial empty positions
        self.goats_placed = 0 # initial number of goats placed is 0
        self.winner =''
        self.done = False
        self.moves_since_last_capture = 0
        self.max_moves_without_capture = 50
        
    def reset(self):
        self.board = np.zeros((5,5), dtype=int) # reset board state
        self.board[0,0] = self.board[0,4] = self.board[4,0] = self.board[4,4] = 1 # place tigers in corners
        self.phase = "Placement" # reset game phase to Placement
        self.turn = -1 # reset turn to goat's turn
        self.goats_killed = 0 # reset number of goats killed
        self.tigers_captured = 0 # reset number of tigers captured
        self.goat_positions = [] # reset goat positions
        self.tiger_positions = [(0,0), (0,4), (4,0), (4,4)] # reset tiger positions
        self.empty_positions = [(i,j) for i in range(5) for j in range(5) if self.board[i,j] == 0] # reset empty positions
        self.goats_placed = 0 # reset number of goats placed
        self.done = False
        self.winner = ''
        self.previous_states = [self.board.copy()]
        return self.board

    def baghchal_reward(self, state):
            """
            Calculates the reward for a given state of the Baghchal game.
            
            Parameters:
            state (numpy.ndarray): The current state of the game.
            
            Returns:
            tuple: A tuple (reward_tigers, reward_goats) representing the rewards for the tigers and goats players, respectively.
            """
            if self.done:
                if self.winner == 'T':
                    return(1,-1)
                elif self.winner=='G':
                    return(-1,1)
                elif self.winner == 'D':
                    return(0,0)
            else:
                return(0,0)

    def step(self, action):
        i,j = action
        if self.turn == -1: # goat's turn
            if self.phase == "Placement":
                if self.goats_placed == 20: # change game phase to movement phase
                    self.phase = "Movement"
                if action not in self.empty_positions: # invalid move
                    return self.board, (0,-0.5), False, {} #punish goat for invalid move  with -0.5 reward
                self.board[i,j] = -1 # place goat object
                self.empty_positions.remove(action) # remove empty position
                self.goat_positions.append(action) # add goat position
                self.goats_placed += 1 # increment number of goats placed
                self.turn = 1
            else: # Movement phase
                if action not in self.empty_positions: # invalid move
                    return self.board, (0,-0.5), False, {} #punish goat for invalid move  with -0.5 reward
                for goat in self.goat_positions:
                    a,b = goat
                    if action in self.get_valid_moves(goat): # move goat object
                        self.board[i,j] = -1
                        self.board[a,b] = 0
                        self.goat_positions.remove(goat)
                        self.goat_positions.append(action)
                        if self.check_tiger_capture(action): # check for tiger capture
                            self.tigers_captured += 1
                            self.tiger_positions.remove(self.get_tiger_capture_position(action))
                        self.turn = 1 # change turn to tiger's turn
                        break
                else: # invalid move
                    return self.board, (0,-0.5), False, {} #punish goat for invalid move  with -0.5 reward
        else: # tiger's turn
            if self.check_goat_jump(): # check for goat jump situation
                valid_moves = self.get_valid_moves_with_goat_jump()
            else:
                valid_moves = self.get_valid_moves_without_goat_jump()
            if action not in valid_moves: # invalid move
                return self.board, (-0.5,0), False, {} #punish tiger for invalid move  with -0.5 reward
            self.board[i,j] = 1 # move tiger object
            self.empty_positions.append(self.tiger_positions[0]) # add empty position
            self.tiger_positions.pop(0) # remove tiger position
            if self.check_goat_capture(action): # check for goat capture
                self.goats_killed += 1
                self.goat_positions.remove(self.get_goat_capture_position(action))
                self.empty_positions.append(self.get_goat_capture_position(action))
            self.turn = -1 # change turn to goat's turn
            #return self.board, (0,0), False, {} # valid move

        # Check if the game is over after the move
        if self.is_game_over():
            self.previous_states.append(self.board.copy())
            reward_tigers, reward_goats = self.baghchal_reward(self.board)
            return self.board.copy(), (reward_tigers, reward_goats), self.done, {}

        
        # Update the previous states list
        self.previous_states.append(self.board.copy())
        
        # Check if the game is a draw due to stalemate
        if self.moves_since_last_capture >= self.max_moves_without_capture:
            self.done = True
            self.winner = 'D'
            reward_tigers, reward_goats = self.baghchal_reward(self.board)
            return self.board.copy(), (reward_tigers, reward_goats), self.done, {}
            #return self.board.copy(),(0,0),self.done,{}
        
        # Return the board, reward, done flag and an empty dictionary
        reward_tigers, reward_goats = self.baghchal_reward(self.board)
        return self.board.copy(), (reward_tigers, reward_goats), self.done, {}

    

    def _rgb_array(self):
        """
        Return a numpy array representing the RGB image of the current state of the game.
        """
        board_rgb = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        board_rgb[self.board == 1] = np.array([255, 0, 0])  # tigers positions marked with red color
        board_rgb[self.board == -1] = np.array([255, 255, 255])  # goats positions marked with white color
        board_rgb = np.rot90(board_rgb)
        return board_rgb

    def render(self, mode='human'):
        if mode == 'human':
            print(self.board)
        elif mode == 'ascii':
            print('  ' + ' '.join(str(i) for i in range(self.board_size)))
            for i in range(self.board_size):
                row = ''.join('.' if self.board[i, j] == 0 else 'T' if (i, j) in self.tigers_positions else 'G' for j in range(self.board_size))
                print(i, row)
        elif mode == 'rgb_array':
            return self._rgb_array()
        else:
            super(BaghchalEnv, self).render(mode=mode)


    def close(self):
        pass


    def is_game_over(self):
        # if len(self.goats_positions) == 0:
        #     self.done = True
        #     return True

        # Check if tigers have captured 5 goats
       # if len(self.tigers_positions) == 0 or (self.num_goats - len(self.goats_positions)) >= 5:
        if (self.goats_killed >= 5):
            self.done = True
            self.winner = 'T'
            return True

        # Check if goats have blocked tigers from being able to move
        for i, j in self.tiger_positions:
            valid_moves = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            valid_moves = [(p, q) for p, q in valid_moves if 0 <= p < self.board_size and 0 <= q < self.board_size]
            for p, q in valid_moves:
                if self.board[p, q] == 0:
                    return False
            self.done = True
            self.winner = 'G'
            return True



    def get_valid_moves(self, position):
        valid_moves = []
        i, j = position
        if i > 0 and self.board[i-1, j] == 0:
            valid_moves.append((i-1, j))
        if i < 4 and self.board[i+1, j] == 0:
            valid_moves.append((i+1, j))
        if j > 0 and self.board[i, j-1] == 0:
            valid_moves.append((i, j-1))
        if j < 4 and self.board[i, j+1] == 0:
            valid_moves.append((i, j+1))
        return valid_moves

    def check_tiger_capture(self, position):
        i, j = position
        if (i > 1 and self.board[i-1, j] == 1 and self.board[i-2, j] == -1) or \
        (i < 3 and self.board[i+1, j] == 1 and self.board[i+2, j] == -1) or \
        (j > 1 and self.board[i, j-1] == 1 and self.board[i, j-2] == -1) or \
        (j < 3 and self.board[i, j+1] == 1 and self.board[i, j+2] == -1):
            return True
        return False

    def get_tiger_capture_position(self, position):
        """
        Returns the position of the goat that the tiger can capture.
        If there is no goat that the tiger can capture, returns None.

        Parameters:
            position (tuple): the position of the tiger

        Returns:
            tuple: the position of the goat to be captured or None
        """
        i,j = position
        # check if the given position is a tiger
        if self.board[i,j] != 1:
            return None
        
        # iterate over all possible positions for the goat to be captured
        for capture_position in self.goat_positions:
            if capture_position == position:
                continue # cannot capture self
            # check if the goat is adjacent to the tiger
            if (capture_position[0] == position[0] and abs(capture_position[1] - position[1]) == 2) or \
            (capture_position[1] == position[1] and abs(capture_position[0] - position[0]) == 2):
                # check if there is no goat between the tiger and the goat to be captured
                in_between_position = ((position[0] + capture_position[0]) // 2, (position[1] + capture_position[1]) // 2)
                if self.board[in_between_position] == -1:
                    return capture_position
        
        # no goat to be captured
        return None


    def check_goat_capture(self, position):
        i, j = position
        if ((i > 0 and self.board[i-1, j] == 1 and self.board[i-2, j] == -1) or
        (i < 4 and self.board[i+1, j] == 1 and self.board[i+2, j] == -1) or
        (j > 0 and self.board[i, j-1] == 1 and self.board[i, j-2] == -1) or
        (j < 4 and self.board[i, j+1] == 1 and self.board[i, j+2] == -1)):
            return True
        return False

    def get_goat_capture_position(self, position):
        i, j = position
        if i > 1 and self.board[i-1, j] == 1 and self.board[i-2, j] == -1:
            return (i-1, j)
        if i < 3 and self.board[i+1, j] == 1 and self.board[i+2, j] == -1:
            return (i+1, j)
        if j > 1 and self.board[i, j-1] == 1 and self.board[i, j-2] == -1:
            return (i, j-1)
        if j < 3 and self.board[i, j+1] == 1 and self.board[i, j+2] == -1:
            return (i, j+1)
        return None

    def check_goat_jump(self):
        return True if self.goats_placed >= 20 else False

    def get_valid_moves_with_goat_jump(self):
        valid_moves = []
        for position in self.goat_positions:
            valid_moves.extend(self.get_valid_moves(position))
        return valid_moves

    def get_valid_moves_without_goat_jump(self):
        valid_moves = []
        for position in self.goat_positions:
            valid_moves.extend(self.get_valid_moves(position))
        valid_moves = list(set(valid_moves))
        invalid_moves = [self.get_goat_capture_position(position) for position in self.goat_positions if self.check_goat_capture(position)]
        invalid_moves = list(filter(lambda x: x is not None, invalid_moves))
        valid_moves = list(set(valid_moves) - set(invalid_moves))
        return valid_moves

#The `__init__` method initializes the environment, with the board size, number of tigers, and number of goats. The board is represented as a numpy array of integers, where 0 means empty, 1 means tiger, and -1 means goat. The `turn` variable keeps track of whose turn it is (tigers or goats), and the `done` variable is set to `True` when the game is over.

#The `reset` method resets the environment to its initial state, with the tigers and goats positioned in their starting positions.

#The `step` method receives an action (a tuple of integers representing a row and a column), and updates the board state accordingly. If the action is invalid (e.g. trying to move to an occupied cell), no change is made to the board. If it is a valid move, and the tigers can eat a goat, they do so and receive a reward of 1. If a goat reaches the other side of the board, the game is over and the goats win with a reward of 10.

#The `render` method can display the board in either "human" or "ascii" mode. In "human" mode, it prints the numpy array to the console. In "ascii" mode, it prints a more visually appealing representation of the board.

#The `close` method is empty, as it is not needed for this environment.