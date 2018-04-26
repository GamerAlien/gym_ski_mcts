import MCTS as mcts
import time
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class XO(mcts.Game):
    def __init__(self):
        board = np.zeros((3, 3))
        super().__init__(board)
        self.reward = 0
        self.XX = 1
        self.OO = 2
        self.EMPTY = 0
        self.turn_tracker = True  # true = playerX, false = playerO
        self.move_stack = []
        self.is_over = False

    def reset(self):
        self.board = np.zeros((3, 3))
        self.turn_tracker = True  # true = playerX, false = playerO
        self.move_stack = []
        self.is_over = False
        self.reward = 0

    def reset_training(self):
        self.learning_game_states.clear()
        self.learning_targets.clear()

    # def get_state(self):
    #     outarray = np.zeros((4, 9))
    #     for i in range(3):
    #         condition = self.board == i
    #         outarray[i] = condition.astype(int).flatten()
    #     if self.turn_tracker:
    #         outarray[3] = np.ones((1, 9))
    #     else:
    #         outarray[3] = np.zeros((1, 9))
    #     return outarray.flatten()

    def get_state(self):
        outarray = np.zeros((3, 9))
        for i in range(3):
            condition = self.board == i
            outarray[i] = condition.astype(int).flatten()
        return np.append(outarray.flatten(), int(self.turn_tracker))

    def possible_moves(self):
        if self.is_over:
            return np.array([])
        return np.array(np.where(self.board == 0)).T

    def make_move(self, action):
        if not self.is_over:
            if self.turn_tracker and not self.is_over:
                self.board[action[0], action[1]] = self.XX
                self.move_stack.append(action)
            elif not self.turn_tracker and not self.is_over:
                self.board[action[0], action[1]] = self.OO
                self.move_stack.append(action)
            self.turn_tracker = not self.turn_tracker
        self.check_win()

    def unmake_move(self):
        self.is_over = False
        last_action = self.move_stack.pop()
        self.board[last_action[0], last_action[1]] = self.EMPTY
        self.turn_tracker = not self.turn_tracker

    def ai_move(self, net):
        next_state_values = []
        possible_moves = self.possible_moves()
        for move_index in range(possible_moves.shape[0]):
            move = possible_moves[move_index]
            self.make_move(move)
            print(self.board)
            move_eval = net(Variable(torch.cuda.FloatTensor(self.get_state())))
            print(move_eval)
            next_state_values.append(move_eval.data[0])
            self.unmake_move()
        if self.turn_tracker:
            ai_move = np.argmax(next_state_values)
        else:
            ai_move = np.argmin(next_state_values)
        self.make_move(self.possible_moves()[ai_move])
        print(self.board)

    def check_win(self):
        #check rows
        for i in range(self.board.shape[0]):
            if np.where(self.board[i] == self.XX)[0].shape[0] == 3:
                self.is_over = True
                self.reward = 1
            elif np.where(self.board[i] == self.OO)[0].shape[0] == 3:
                self.is_over = True
                self.reward = -1
        #check cols
        for i in range(self.board.shape[0]):
            if np.where(self.board.T[i] == self.XX)[0].shape[0] == 3:
                self.is_over = True
                self.reward = 1
            elif np.where(self.board.T[i] == self.OO)[0].shape[0] == 3:
                self.is_over = True
                self.reward = -1
        if self.board[0, 0] == self.XX and self.board[1, 1] == self.XX and self.board[2, 2] == self.XX:
            self.is_over = True
            self.reward = 1
        elif self.board[0, 0] == self.OO and self.board[1, 1] == self.OO and self.board[2, 2] == self.OO:
            self.is_over = True
            self.reward = -1
        elif self.board[0, 2] == self.XX and self.board[1, 1] == self.XX and self.board[2, 0] == self.XX:
            self.is_over = True
            self.reward = 1
        elif self.board[0, 2] == self.OO and self.board[1, 1] == self.OO and self.board[2, 0] == self.OO:
            self.is_over = True
            self.reward = -1
        if np.array(np.where(self.board == 0)).size == 0:
            self.is_over = True
            self.reward = 0

    def get_endgame_reward(self):
        return self.reward


def __main__():
    playout_policy = mcts.RandomPlayout()
    game = XO()
    initial_root = mcts.Nodemcts(None)
    mcts_xo = mcts.mcts(game, playout_policy, initial_root)
    mcts_xo.run(10000)

    node = mcts_xo.find_node(list(mcts_xo.game.move_stack))
    while True:
        input("press key to continue...")
        node = mcts_xo.greedy_move(node)



def _test_get_state():
    game = XO()
    print(game.get_state())


def _test_win():
    game = XO()
    game.board = np.array([[0, 2, 0], [0, 2, 0], [0, 2, 0]])
    game.check_win()
    print(game.is_over, game.reward)

# def _test_mcts():
#     playout_policy = mcts.RandomPlayout()
#     game = XO()
#     initial_root = mcts.Nodemcts(None)
#     mcts_xo = mcts.mcts(game, playout_policy, initial_root)
#     node1 = mcts.Nodemcts(initial_root)
#     node2 = mcts.Nodemcts(initial_root)
#     node3 = mcts.Nodemcts(initial_root)
#     mcts_xo.backup(1, node3)

# start_time = time.time()
# __main__() #for testing
# _test_get_state()  # for testing
# print("--- %s seconds ---" % (time.time() - start_time))

# _test_win()
# _test_mcts()