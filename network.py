import numpy as np
from itertools import count
from collections import namedtuple
import tictactoe as ttt
import MCTS as mcts
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class MCTSNetwork(nn.Module):
    # def __init__(self, input_size, action_space):
    def __init__(self, input_size):
        super(MCTSNetwork, self).__init__()
        # input of state (s), output of move probabilities
        # p with components p_a=Pr(a|s)
        # and scalar value v estimating expected outcome z from position s
        # loss function l =(z-v)**2 - pi log(p) + c||theta||**2
        # (p,v) = f_theta(s)
        # c controls the level of L2 regularization
        self.layer_one = nn.Linear(input_size, 1024)
        self.layer_two = nn.Linear(1024, 256)
        self.layer_three = nn.Linear(256, 256)
        self.layer_four = nn.Linear(256, 32)
        # self.layer_four = nn.Linear(256, 256)
        # self.layer_five = nn.Linear(256, 256)
        # self.layer_six = nn.Linear(256, 32)

        # self.action_layer = nn.Linear(128, 64)
        # self.action_head = nn.Linear(64, action_space)

        self.value_layer = nn.Linear(32, 32)
        self.value_head = nn.Linear(32, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.layer_one(x))
        x = F.relu(self.layer_two(x))
        x = F.relu(self.layer_three(x))
        x = F.relu(self.layer_four(x))
        # x = F.relu(self.layer_five(x))
        # x = F.relu(self.layer_six(x))
        # ax = F.relu(self.action_layer(x))
        vx = F.relu(self.value_layer(x))

        # action_scores = self.action_head(ax)
        state_values = self.value_head(vx)
        # return F.softmax(action_scores, dim=-1), state_values
        return state_values


def training_loop(learning_rate, num_iters, mcts_iters, nn_size=1, net=None, optimizer=None, mcts_initial=0):
    ## init game
    env = ttt.XO()
    initial_root = mcts.Nodemcts(None)
    playout_policy = mcts.RandomPlayout()
    mcts_xo = mcts.mcts(env, playout_policy, initial_root)
    ## nnet ingredients
    if net is None and optimizer is None:
        net, optimizer = init_net(nn_size, learning_rate)
    mcts_xo.run(mcts_initial)
    mcts_xo.game.reset_training()
    loss_samples = []
    for _ in tqdm(range(num_iters)):
        optimizer.zero_grad()
        net.zero_grad()
        gamestates, targets = mcts_xo.run(mcts_iters)
        train_net(net, optimizer, gamestates, targets, loss_samples)
    plt.plot(loss_samples)
    plt.show()
    return net, mcts_xo, env


def init_net(input_size, learning_rate):
    net = MCTSNetwork(input_size)
    optimizer = optim.SGD(net.parameters(), learning_rate)
    net.cuda()
    return net, optimizer


def train_net(net, optimizer, input, targets, error_samples):
    optimizer.zero_grad()
    if input:
        net_input = Variable(torch.cuda.FloatTensor(input))
        net_targets = Variable(torch.cuda.FloatTensor(targets))
        output = net(net_input)
        loss = net.loss_fn(output, net_targets)
        error_samples.append(loss.data[0])
        loss.backward()
        optimizer.step()
    return error_samples
# training_loop(0.01, 1)
