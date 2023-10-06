from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from modes import ModeController
from sprites import GhostSprites

import torch

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import namedtuple, deque

import random

AI = True

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

"""
As of gamm 0.99 or 0.95 seem to work just fine.
EPS_START can vary between 1 and 0.9 without much difference
EPS_END is not that important in this scenario, the DQN usually learns sooner if it learns at all
EPS_DECAY is the most important parameter, it really changes the speed of learning, the higher the slower, but the slower the more it learns
TAU is not that important, it can be 0.001 or 0.0001, it just changes the speed of learning, the higher the slower, but the slower the more it learns
LR seem to work fine with 1e-3, 1e-4 seem to lead to very inconsistent results
"""

GAMMA = 0.99  # discount factor, greater means more importance to future rewards
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 100_000
TAU = 0.001  # for soft update of target parameters, so greater means more soft
LR = 1e-3  # learning rate, greater means faster learning

BATCH_SIZE = 64  # 128 is suggested batch size for DQN

steps_done = 0

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# at all time we have 1 acton per ghost, so 2 actions, (blinky and pinky)
# the possible actions are 4 for each ghost, so 16 possible actions
# to reconstruct the action:
"""
0 = blinky up, pinky up
1 = blinky up, pinky down
2 = blinky up, pinky left
3 = blinky up, pinky right
4 = blinky down, pinky up
5 = blinky down, pinky down
6 = blinky down, pinky left
7 = blinky down, pinky right
8 = blinky left, pinky up
9 = blinky left, pinky down
10 = blinky left, pinky left
11 = blinky left, pinky right
12 = blinky right, pinky up
13 = blinky right, pinky down
14 = blinky right, pinky left
15 = blinky right, pinky right
"""
# the observation space is the position of the pacman and the ghosts
# the observation space is 2 for each ghost, so 4
# the observation space is 2 for the pacman, so 2
# so the observation space is 6
n_actions = 16
n_obs = 6


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, game, dt):
    global steps_done
    sample = random.random()
    if game.training:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
    else:
        eps_threshold = 0.00
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Calculate Q-values for all actions
            q_values = policy_net(state)

            # Remove invalid actions by masking them with a large negative value
            # for i in range(n_actions):
            #     if not isValidAction(i, game, dt):
            #         q_values[0][i] = -float('inf')

            # Select the action with the highest Q-value
            action = q_values.max(1)[1].view(1, 1)

            return action
    else:
        return randomAction(state, game, dt)


def randomAction(state, game, dt):
    # random action
    action = random.randrange(n_actions)
    # possibleActions = list(range(n_actions))
    # while not isValidAction(action, game, dt):
    #     # Calculate Q-values for all actions
    #     q_values = policy_net(state)

    #     q_values[0][action] = -float('inf')

    #     possibleActions.remove(action)
    #     action = random.choice(possibleActions)

    return torch.tensor([[action]], device=device, dtype=torch.long)


def isValidAction(action, game, dt):
    # basically if the action makes the ghost collide with a wall, it is invalid
    # we need to convert the action to the 2 actions for each ghost, remembering that -2 = right, -1 = left, 1 = up, 2 = down
    action1, action2 = getAction(action)

    if game.ghosts.pinky.willOvershotTarget(action1, dt) or game.ghosts.blinky.willOvershotTarget(action2, dt):
        return False

    return True


def getAction(action):
    action1 = STOP
    action2 = STOP

    if action < 4:
        action1 = UP
    elif action < 8:
        action1 = DOWN
    elif action < 12:
        action1 = LEFT
    elif action < 16:
        action1 = RIGHT

    if action % 4 == 0:  # 0, 4, 8, 12
        action2 = UP
    elif action % 4 == 1:  # 1, 5, 9, 13
        action2 = DOWN
    elif action % 4 == 2:
        action2 = LEFT
    elif action % 4 == 3:
        action2 = RIGHT

    return action1, action2


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        return self.layer4(x)


policy_net = DQN(n_observations=n_obs, n_actions=n_actions).to(device)
target_net = DQN(n_observations=n_obs, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# 10000 is suggested for memory, seems like a lot
memory = ReplayMemory(1000000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]
    # compute the final state values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping, to prevent the gradient from exploding, we clip the gradient to a maximum value of 1
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)

    optimizer.step()


class Ghost(Entity):
    def __init__(self, node, pacman=None, blinky=None):
        Entity.__init__(self, node)
        self.name = GHOST
        self.points = 200
        if not AI:
            self.goal = Vector2()
        if not AI:
            self.directionMethod = self.goalDirection
        self.pacman = pacman
        self.mode = ModeController(self)
        self.blinky = blinky
        self.homeNode = node
        self.choice = STOP
        self.cell = None
        self.lastPosition = None

    def reset(self):
        Entity.reset(self)
        self.points = 200
        if not AI:
            self.directionMethod = self.goalDirection

    def findCell(self, game):
        # assign cell to ghost (the cell is the closest to the ghost)
        cell = None
        d = 100000000
        for c in game.cells.cellDict.values():
            dist = (c.position - self.position).magnitudeSquared()
            if dist < d:
                d = dist
                cell = c
        self.cell = cell

    def update(self, dt):
        self.sprites.update(dt)
        self.mode.update(dt)

        if not AI or self.mode.current is SPAWN or self.mode.current is FREIGHT:
            if self.mode.current is SCATTER:
                self.scatter()
            elif self.mode.current is CHASE:
                self.chase()
            Entity.update(self, dt)
        else:
            direction = self.choice
            self.position += self.directions[self.direction]*self.speed*dt
            if self.overshotTarget():
                self.node = self.target
                if self.node.neighbors[PORTAL] is not None:
                    self.node = self.node.neighbors[PORTAL]
                self.target = self.getNewTarget(direction)
                if self.target is not self.node:
                    self.direction = direction
                else:
                    self.target = self.getNewTarget(self.direction)

                if self.target is self.node:
                    self.direction = STOP
                self.setPosition()
            else:
                if self.oppositeDirection(direction):
                    self.reverseDirection()

    def scatter(self):
        print("Scatter not implemented")
        self.goal = Vector2()

    def chase(self):
        self.goal = self.pacman.position

    def spawn(self):
        self.goal = self.spawnNode.position

    def setSpawnNode(self, node):
        self.spawnNode = node

    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current == SPAWN:
            self.setSpeed(SPEED * 1.5)
            if not AI:
                self.directionMethod = self.goalDirection
            self.spawn()

    def startFreight(self):
        self.mode.setFreightMode()
        if self.mode.current == FREIGHT:
            self.setSpeed(SPEED / 2)
            self.directionMethod = self.randomDirection

    def normalMode(self):
        self.setSpeed(SPEED)
        if not AI:
            self.directionMethod = self.goalDirection
        self.homeNode.denyAccess(DOWN, self)


class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = BLINKY
        self.color = RED
        self.sprites = GhostSprites(self)


class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = PINKY
        self.color = PINK
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILEWIDTH*NCOLS, 0)

    def chase(self):
        self.goal = self.pacman.position + \
            self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4


class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = INKY
        self.color = TEAL
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILEWIDTH*NCOLS, TILEHEIGHT*NROWS)

    def chase(self):
        vec1 = self.pacman.position + \
            self.pacman.directions[self.pacman.direction] * TILEWIDTH * 2
        vec2 = (vec1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vec2


class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = CLYDE
        self.color = ORANGE
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(0, TILEHEIGHT*NROWS)

    def chase(self):
        d = self.pacman.position - self.position
        ds = d.magnitudeSquared()
        if ds <= (TILEWIDTH * 8)**2:
            self.scatter()
        else:
            self.goal = self.pacman.position + \
                self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4


class GhostGroup(object):
    def __init__(self, node, pacman, training):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        # self.inky = Inky(node, pacman, self.blinky)
        # self.clyde = Clyde(node, pacman)
        # self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]
        self.ghosts = [self.blinky, self.pinky]
        self.total_reward = 0
        self.frames = 0
        self.resets = 0
        self.rewards = []
        self.action = None
        self.state = None

        self.training = training
        if not self.training:
            policy_net.load_state_dict(torch.load('ghosts.pt'))
            policy_net.eval()

    def getState(self, game):
        return torch.tensor(game.getState(), dtype=torch.float32, device=device).unsqueeze(0)

    def getMoves(self, game, dt):
        if self.action == None:
            self.blinky.findCell(game)
            self.pinky.findCell(game)
        state = self.getState(game)
        self.state = state
        action = select_action(state, game, dt)
        self.action = action

        # we need to convert the action to the 2 actions for each ghost, remembering that -2 = right, -1 = left, 1 = up, 2 = down
        # and we have 4 possible actions for each ghost, so 4*4 = 16 possible actions, from 0 to 15
        # so we can convert the action to the 2 actions for each ghost by doing:

        self.blinky.choice, self.pinky.choice = getAction(action)

        self.blinky.findCell(game)
        self.pinky.findCell(game)

    def completeTraining(self, game, dt):
        if self.action is None:
            return
        terminated = game.terminated

        # closest cell to pacman is the pacman cell
        if not game.pacman.cell:
            closest_cell = None
            pacman_cell_distance = 100000000
            for cell in game.cells.cellDict.values():
                d = cell.position - game.pacman.position
                dSquared = d.magnitudeSquared()
                if dSquared < pacman_cell_distance:
                    pacman_cell_distance = dSquared
                    closest_cell = cell

            game.pacman.cell = closest_cell

        # we want to try maximize speed
        # reward is the inverso of the sum of the distances of the ghosts to the pacman
        blinkyDistance = game.cells.getBFSDistance(
            game.ghosts.blinky.cell, game.pacman.cell)
        pinkyDistance = game.cells.getBFSDistance(
            game.ghosts.pinky.cell, game.pacman.cell)
        if pinkyDistance == -1:
            pinkyDistance = 20  # 20 is the starting distance
        if blinkyDistance == -1:
            blinkyDistance = 20  # 20 is the starting distance

        distance = pinkyDistance + blinkyDistance
        if distance < 20:
            reward = 20 - distance
        else:
            reward = -(distance**2)/100

        # we also give less reward if the ghosts did not move
        if self.pinky.lastPosition and self.pinky.position == self.pinky.lastPosition:
            reward -= 50
        if self.blinky.lastPosition and self.blinky.position == self.blinky.lastPosition:
            reward -= 50

        self.blinky.lastPosition = self.blinky.position
        self.pinky.lastPosition = self.pinky.position

        if game.pacman.dead:
            reward = 100
            game.pacman.dead = False
            terminated = True
            self.blinky.lastPosition = None
            self.pinky.lastPosition = None
            print("Pacman died 🪦")
        else:
            if not terminated:
                reward -= 1
            else:
                reward -= 100
                self.blinky.lastPosition = None
                self.pinky.lastPosition = None
                print("Pacman won 😢")

        self.total_reward += reward
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                game.getState(), dtype=torch.float32, device=device).unsqueeze(0)

        if game.debugging:
            print("------------------------")
            print("Reward: ", reward)
            print("Next state: ", next_state)
            if game.ghosts.pinky.cell:
                print("Pinky: ", game.ghosts.pinky.cell.position)
            if game.ghosts.blinky.cell:
                print("Blinky: ", game.ghosts.blinky.cell.position)
            if game.pacman.cell:
                print("Pacman: ", game.pacman.cell.position)
            print("------------------------")

        if self.state is None or self.action is None or reward is None:
            print(self.state, self.action, reward)
            return

        memory.push(self.state, self.action, next_state, reward)

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            game.terminated = False
            self.resets += 1
            self.rewards.append(self.total_reward)
            average = sum(self.rewards) / len(self.rewards)
            # we print the results every episode
            print('Episode: ', self.resets, 'Reward: ',
                  self.total_reward, "EPS: ", EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY), "Steps: ", steps_done, "Average: ", average, "Memory: ", len(memory))

            self.total_reward = 0

            if steps_done > EPS_DECAY * 3:
                self.saveModel()
                exit()

    def saveModel(self):
        plt.plot(self.rewards)
        plt.savefig('ghosts_nn.png')
        # save the model
        torch.save(policy_net.state_dict(),
                   'ghosts.pt')

    def __iter__(self):
        return iter(self.ghosts)

    def update(self, dt, game):
        self.frames += dt
        if self.frames >= 0.2:  # 10 frames with dt = 0.02
            self.frames = 0
            if self.training:
                self.completeTraining(game, dt)
            self.getMoves(game, dt)

        for ghost in self:
            ghost.update(dt)

    def startFreight(self):
        for ghost in self:
            ghost.startFreight()
        self.resetPoints()

    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    def updatePoints(self):
        for ghost in self:
            ghost.points *= 2

    def resetPoints(self):
        for ghost in self:
            ghost.points = 200

    def hide(self):
        for ghost in self:
            ghost.visible = False

    def show(self):
        for ghost in self:
            ghost.visible = True

    def reset(self):
        for ghost in self:
            ghost.reset()

    def render(self, screen):
        for ghost in self:
            ghost.render(screen)
