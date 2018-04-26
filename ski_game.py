import gym
import MCTS as mcts
import time
import numpy as np
import copy
import network
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import cProfile
from pickle import dumps, loads

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    # plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    # display(display_animation(anim, default_mode='loop'))
    anim.save('E:/dropbox/Dropbox/Dropbox/McGill/Winter 2018/COMP-767/project/anim_frames/animation.gif',
              writer='imagemagick', fps=30)
    plt.show()


class SkiGame(mcts.Game):
    def __init__(self):
        super().__init__(None)
        self.env = gym.make('Skiing-ram-v0')
        self.env.reset()
        # self.env = self.env.unwrapped
        self.is_over = False
        self.move_stack = []

    def render(self):
        self.env.render()

    def show_snap(self):
        self.env.render()
        time.sleep(0.8)
        self.env.close()

    def possible_moves(self):
        return [x for x in range(self.env.action_space.n)]

    def action_space(self):
        return self.env.action_space

    def sample_move(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def make_move(self, action):
        ob, reward, done, _ = self.env.step(action)
        self.move_stack.append(action)
        if done:
            self.is_over = True
        return ob, reward, done

    def mcts_simulation(self, playout_policy):
        # game_state = copy.deepcopy(self.env)
        saved_state = self.env.unwrapped.clone_state()
        target_depth = 10
        total_reward = 0
        for _ in range(target_depth):
            obs, reward, done = self.make_move(self.sample_move())
            total_reward += reward
        self.env.unwrapped.restore_state(saved_state)
        return obs, total_reward


class RandomSki(mcts.PlayoutPolicy):
    def get_action(self, possible_moves):
        return possible_moves.sample()


class NodeSki(mcts.Nodemcts):
    def __init__(self, parent, children=None, action=None):
        super().__init__(parent, children, action)

    def backup(self, obs, reward, game):
        self.nsims += 1
        self.nwins += reward
        game.learning_targets.append(reward)
        game.learning_game_states.append(obs)
        if self._parent is not None:
            self._parent.backup(obs, reward, game)

    def full_expansion(self, action_list):
        for action in action_list:
            NodeSki(parent=self, action=action)

    def selection(self, total_plays):
        if not self._children:
            return self
        for child in self._children:
            child[0].update_ucb(total_plays)
        ucbs = np.argmax(np.array([child[0].ucb_mean + child[0].ucb_deviation for child in self._children]))
        return self._children[ucbs]


class SkiMcts(mcts.mcts):
    def __init__(self, game, playout_policy, root_node=None):
        super().__init__(game, playout_policy, root_node)
        self.root_node = NodeSki(None)

    def backup(self, obs, reward, node):
        node.backup(obs, reward, self.game)

    def run(self, max_iters, start_node=None):
        '''
        Performs max_steps number of iterations of MCTS
        :param max_steps: Max tries to attempt before terminating
        '''
        # self.game.reset()
        self.game.reset_training()
        saved_stack = copy.deepcopy(self.game.move_stack)
        # saved_state = copy.deepcopy(self.game.env)
        saved_state = self.game.env.unwrapped.clone_state()
        # frames=[]
        for _ in range(max_iters):
            # self.game.env = copy.deepcopy(saved_state)
            self.game.env.unwrapped.restore_state(saved_state)
            # frames.append(self.game.env.render(mode='rgb_array'))
            if start_node is None:
                node_cursor = self.selection(self.root_node)
            else:
                node_cursor = self.selection(start_node)
            node_cursor.full_expansion(self.game.possible_moves())
            node_cursor = self.selection(node_cursor)
            # frames.append(self.game.env.render(mode='rgb_array'))
            obs, total_reward = self.game.mcts_simulation(self.playout_policy)
            self.backup(obs, total_reward, node_cursor)
        # self.game.env = saved_state
        self.game.env.unwrapped.restore_state(saved_state)
        self.game.move_stack = saved_stack
        # display_frames_as_gif(frames)
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
                if np.equal(child[1], move_stack[-1]):
                    node = child[0]
                    move_stack.pop()
                    break
        return node

    def greedy_move(self, node, render=False):
        nsims = [child[0].nsims for child in node._children]
        next_state = None
        if nsims:
            index = np.argmax(nsims)
            move = node._children[index][1]
            next_state = node._children[index][0]
            self.game.make_move(move)
            if render:
                self.game.env.render()
        return next_state

    def training_loop_ski(learning_rate, num_iters, mcts_iters, nn_size=1, net=None, optimizer=None, mcts_initial=0):
        ## init game
        env = SkiGame()
        playout_policy = RandomSki()
        mcts_ski = SkiMcts(env, playout_policy)
        ## nnet ingredients
        if net is None and optimizer is None:
            net, optimizer = network.init_net(nn_size, learning_rate)
        mcts_ski.run(mcts_initial)
        mcts_ski.game.reset_training()
        loss_samples = []
        for _ in tqdm(range(num_iters)):
            optimizer.zero_grad()
            net.zero_grad()
            gamestates, targets = mcts_ski.run(mcts_iters)
            network.train_net(net, optimizer, gamestates, targets, loss_samples)
        plt.plot(loss_samples)
        plt.show()
        return net, mcts_ski, env


def test():
    game = SkiGame()
    rollout_policy = RandomSki()
    mcts_ski = SkiMcts(game, rollout_policy)
    episodes = 150
    # cProfile.runctx('mcts_ski.run(1)', None, locals())
    # mcts_ski.run(100)
    current_node = mcts_ski.root_node
    frames = []
    for _ in range(episodes):
        # while game.is_over:
        mcts_ski.run(30, current_node)
        current_node = mcts_ski.greedy_move(current_node)
        frames.append(game.env.env.render(mode='rgb_array'))
    display_frames_as_gif(frames)


def save_test():
    env = gym.make('Skiing-ram-v0')
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
    state = env.unwrapped.clone_state()
    time.sleep(2)
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
    time.sleep(2)
    env.unwrapped.restore_state(state)
    time.sleep(2)
    env.render()
    time.sleep(2)
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())


def save_test2():
    env = gym.make('Skiing-ram-v0')
    env.reset()
    frames = []
    for _ in range(100):
        frames.append(env.render(mode='rgb_array'))
        env.step(env.action_space.sample())
    state = env.unwrapped.clone_state()
    for _ in range(100):
        frames.append(env.render(mode='rgb_array'))
        env.step(env.action_space.sample())
    env.unwrapped.restore_state(state)
    for _ in range(100):
        frames.append(env.render(mode='rgb_array'))
        env.step(env.action_space.sample())
    display_frames_as_gif(frames)


def save_test3():
    game = SkiGame()
    game.reset()
    frames = []
    for _ in range(100):
        frames.append(game.env.render(mode='rgb_array'))
        game.env.step(game.env.action_space.sample())
    state = game.env.unwrapped.clone_state()
    for _ in range(100):
        frames.append(game.env.render(mode='rgb_array'))
        game.env.step(game.env.action_space.sample())
    game.env.unwrapped.restore_state(state)
    for _ in range(100):
        frames.append(game.env.render(mode='rgb_array'))
        game.env.step(game.env.action_space.sample())
    display_frames_as_gif(frames)


test()
# save_test()
# save_test3()
