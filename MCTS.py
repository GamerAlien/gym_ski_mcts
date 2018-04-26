import math
import numpy as np
import pdb
from tqdm import tqdm


class Node:
    def __init__(self, parent=None, children=None, action=None):
        if children is not None:
            self._children = children
        else:
            self._children = []
        if parent is not None and action is not None:
            parent.add_child(self, action)
            self._parent = parent
            self.depth = parent.depth + 1
        elif parent is not None:
            self._parent = parent
            self.depth = parent.depth + 1
        else:
            self.depth = 0
            self._parent = None  # root

    def add_child(self, node, action):
        self._children.append((node, action))

    def has_child(self):
        return bool(self._children)


# class Treemcts:
#     def __init__(self, root):
#         if root is not None:
#             self.root_node = root
#         else:
#             self.root_node = Nodemcts(None)
#
#     def selection(self):
#         node = self.root_node
#         while node.has_child():
#             node = node.selection()
#         return node


class Nodemcts(Node):
    def __init__(self, parent, children=None, action=None):
        super().__init__(parent, children, action)
        self.nwins = 0
        self.nsims = 0  # TODO: figure out proper way to do this
        self.board_state = None
        self.ucb_mean = 0
        self.ucb_deviation = 0

    def update_ucb(self, total_plays):
        used_nsims = self.nsims
        used_total_plays = total_plays
        if used_nsims == 0:
            used_nsims = 0.1
        if used_total_plays == 0:
            used_total_plays = 1
        self.ucb_mean = (self.nwins / used_nsims)
        self.ucb_deviation = math.sqrt((2 * math.log(used_total_plays)) / used_nsims)

    def selection(self, total_plays):
        if not self._children:
            return self
        for child in self._children:
            child[0].update_ucb(total_plays)
        if self.depth % 2 == 0:
            ucbs = np.argmax(np.array([child[0].ucb_mean + child[0].ucb_deviation for child in self._children]))
            return self._children[ucbs]
        else:
            ucbs = np.argmin(np.array([child[0].ucb_mean - child[0].ucb_deviation for child in self._children]))
            return self._children[ucbs]

    def single_expansion(self, actions_list):
        selected_action = np.random.choice(actions_list)
        Nodemcts(parent=self, action=selected_action)

    def full_expansion(self, action_list):
        for action in action_list:
            Nodemcts(parent=self, action=action)

    def backup(self, win, game, debug):
        # win_increment = 0
        # if self.depth % 2 == 0 and win == -1:
        #     win_increment = 1
        # if self.depth % 2 == 1 and win == 1:
        #     win_increment = 1
        win_increment = win
        self.nwins += win_increment
        self.nsims += 1

        # # for learning, append the gamestate and targets
        # game.learning_targets.append(win)
        # # game.learning_targets.append(win)
        # game.learning_game_states.append(game.get_state())

        if game.is_over:
            # for learning, append the gamestate and targets
            game.learning_targets.append(win)
            # game.learning_targets.append(win)
            game.learning_game_states.append(game.get_state())

        # unmake last move and move up the chain
        if self._parent is not None:
            game.unmake_move()
            self._parent.backup(win, game, debug)

    def get_target(self):
        '''
        function to return a target to learn towards for this MCTS node.
        :return: target value = wins/sims
        '''

        return float(self.nwins) / float(self.nsims)


class PlayoutPolicy:
    def get_action(self, possible_moves):
        pass


class RandomPlayout(PlayoutPolicy):
    def get_action(self, possible_moves):
        return np.random.choice(possible_moves.shape[0])


class Game:
    def __init__(self, board):
        self.board = board
        self.is_over = False
        self.learning_targets = []
        self.learning_game_states = []

    def possible_moves(self):
        return []

    def reset(self):
        pass

    def reset_training(self):
        pass

    def make_move(self, action):
        pass

    def unmake_move(self):
        pass

    def get_state(self):
        pass

    def get_endgame_reward(self):
        return 0

    def mcts_simulation(self, playout_policy):
        # sim_board = self.board.copy()
        num_moves = 0
        while not self.is_over:
            # print(self.board)
            move_index = playout_policy.get_action(self.possible_moves())
            self.make_move(self.possible_moves()[move_index])
            num_moves += 1
        # print(self.board)
        reward = self.get_endgame_reward()
        for _ in range(num_moves):
            self.unmake_move()
        return reward


class mcts:
    def __init__(self, game, playout_policy, root_node):
        self.game = game
        self.playout_policy = playout_policy
        self.root_node = root_node
        self.total_plays = 0

    def selection(self, node, debug=False):
        while node.has_child() and not self.game.is_over:
            selection = node.selection(self.total_plays)
            node = selection[0]
            action = selection[1]
            if debug is True:
                print(action, node.nwins, node.nsims)
            self.game.make_move(action)
        return node

    def backup(self, reward, node, debug):
        node.backup(reward, self.game, debug)
        self.total_plays += 1

    def run(self, max_steps, start_node=None):
        '''
        Performs max_steps number of iterations of MCTS
        :param max_steps: Max tries to attempt before terminating
        '''
        self.game.reset()
        self.game.reset_training()
        for _ in range(max_steps):
            self.game.reset()
            # self.game.reset_training()
            if start_node is None:
                node_cursor = self.selection(self.root_node)
            else:
                node_cursor = start_node
            node_cursor.full_expansion(self.game.possible_moves())
            node_cursor = self.selection(node_cursor)
            debug = False
            # if self.game.is_over:
            #     debug=True
            reward = self.game.mcts_simulation(self.playout_policy)
            self.backup(reward, node_cursor, debug)
            # self.root_node.update_ucb(self.total_plays)
        return self.game.learning_game_states, self.game.learning_targets

    def run_once(self):
        '''
        Performs a single step of MCTS
        :param max_steps: Max tries to attempt before terminating
        :return learning targets and gamestates for value net
        '''
        self.game.reset()
        node_cursor = self.selection(self.root_node)
        node_cursor.full_expansion(self.game.possible_moves())
        node_cursor = self.selection(node_cursor)
        reward = self.game.mcts_simulation(self.playout_policy)
        self.backup(reward, node_cursor)
        # self.root_node.update_ucb(self.total_plays)
        return self.game.learning_game_states, self.game.learning_targets

    def find_node(self, move_stack):
        """
        pass a copy of a movestack to this!!!
        :param move_stack:
        :return:
        """
        node = self.root_node
        while move_stack:
            for child in node._children:
                if np.equal(child[1], move_stack[-1])[0]:
                    node = child[0]
                    move_stack.pop()
                    break
        return node

    def greedy_move(self, node):
        nsims = [child[0].nsims for child in node._children]
        nwins = [child[0].nwins for child in node._children]
        print(nsims)
        print(nwins)
        next_state = None
        if nsims:
            index = np.argmax(nsims)
            move = node._children[index][1]
            next_state = node._children[index][0]
            self.game.make_move(move)
        return next_state
