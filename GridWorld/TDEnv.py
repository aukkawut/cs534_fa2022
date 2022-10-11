import numpy as np
import gym
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import os
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle
def generate_random_world(size,pw = 0.2, prp = 0.2, prn = 0.2):
    '''
    This function generates a random gridworld.
    size: size of the gridworld
    pw: probability of a wall in the gridworld
    prp: probability of a positive reward in the gridworld
    prn: probability of a negative reward in the gridworld
    '''
    grid = np.zeros((size+1, size+1), dtype='object')
    #make sure that outside the grid is a wall
    grid[0, :] = 'X'
    grid[-1, :] = 'X'
    grid[:, 0] = 'X'
    grid[:, -1] = 'X'
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if np.random.random() < pw:
                grid[i, j] = 'X'
            elif np.random.random() < prp:
                grid[i, j] = str(np.random.randint(1, 9))
            elif np.random.random() < prn:
                grid[i, j] = str(-np.random.randint(1, 9))
            else:
                grid[i, j] = '0'
    grid[size-1, 1] = 'S'
    return grid
class GridWorld(gym.Env):
    '''
    This class is the gridworld class.
    The grid will be read from the tab delimited file.
        S: represents where the agent will start
        X: represents the wall
        0: represents an empty square
        [-9,9]: represents the reward for the square
    There can be multiple goal as there can be multiple rewards

    The agent can move up, down, left and right. If it hits the wall, it will not move.
    The goal is to maximize the reward until the time limit is reached.

      -  The agent has the specific probability p to move one step in the desired direction.
      -  The agent has the probability (1-p)/2 to move two step in the desired direction.
      -  The agent has the probability (1-p)/2 to move one step in the backward direction.

    The agent will get the reward for the square it is in. However, if it moves two steps
    the reward will be the tile it ends up in.
    '''
    def __init__(self,p,r,gridFile) -> None:
        self.grid = []
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Discrete(1)
        self.start = None
        self.action = None
        self.envAction = None
        self.current = None
        self.reward = 0
        self.done = False
        self.p = p
        self.r = r
        self.read_grid(gridFile)
        self.reset()
        
    def read_grid(self,gridfile) -> None:
        '''
        This function reads the grid from the file.
        '''
        try:
            for line in gridfile:
                self.grid.append(line.strip().split('\t'))
            self.grid = np.array(self.grid)
        except:
            self.grid = generate_random_world(10)
        self.observation_space = gym.spaces.Discrete(self.grid.shape[0] * self.grid.shape[1])
        self.start = np.argwhere(self.grid == 'S')[0]
        self.current = self.start
    def reset(self) -> None:
        '''
        This function resets the environment.
        '''
        self.current = self.start
        self.reward = 0
        self.done = False
        self.action = None
        self.envAction = None
        return self.current[0] * self.grid.shape[1] + self.current[1]
    def step(self, action: int) -> tuple:
        '''
        This function takes the action and returns the next state, reward and done.
        
        The agent can move up, down, left and right. If it hits the wall, it will not move.
        The goal is to maximize the reward until the time limit is reached.

        -  The agent has the specific probability p to move one step in the desired direction.
        -  The agent has the probability (1-p)/2 to move two step in the desired direction.
        -  The agent has the probability (1-p)/2 to move one step in the backward direction.
        '''
        #if done, you can't interact with the environment
        self.action = action
        if self.done:
            return self.current[0] * self.grid.shape[1] + self.current[1], self.reward, self.done, self.done,{}
        if np.random.random() < self.p:
            self.envAction = "Normal"
            self.current = self.move(self.current, action)
        elif np.random.random() < 0.5:
            self.envAction = "Double"
            self.current = self.move(self.current, action)
            self.current = self.move(self.current, action)
        else:
            self.envAction = "Backward"
            self.current = self.move(self.current, (action + 2)%4)
        if str(self.grid[self.current[0], self.current[1]]) != '0' and str(self.grid[self.current[0], self.current[1]]) !='S':
            self.done = True
            if str(self.grid[self.current[0], self.current[1]]) == 'S':
                self.reward += self.r
            else:
                self.reward += (int(self.grid[self.current[0], self.current[1]])+ self.r)
        else:
            try:
                self.reward += (int(self.grid[self.current[0], self.current[1]]) + self.r)
            except:
                self.reward += self.r
        return self.current[0] * self.grid.shape[1] + self.current[1], self.reward, self.done, self.done,{}
    def render(self) -> None:
        '''
        This function renders the gridworld.
        '''
        grid = self.grid.copy()
        #highlight the current position
        grid[self.current[0], self.current[1]] = 'A'
        #print the grid out with ANSI colors
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 'X':
                    print('\033[1;31m' + str(grid[i, j]) + '\033[0m', end='\t')
                elif grid[i, j] == 'A':
                    print('\033[1;32m' + str(grid[i, j]) + '\033[0m', end='\t')
                elif grid[i, j] == 'S':
                    print('\033[1;34m' + str(grid[i, j]) + '\033[0m', end='\t')
                else:
                    print(grid[i, j], end='\t')
            print()
        #print position,action taken, reward,action, action taken by the environment
        print(f'Position: {self.current}')
        try:
            print(f'Action: {self.actions[self.action]}')
        except:
            pass
        print(f'Reward: {self.reward}')
        print(f'Action taken by the environment: {self.envAction}')
    def close(self) -> None:
        pass
    def move(self, current: np.ndarray, action: int) -> np.ndarray:
        '''
        This function moves the agent one step in the desired direction.
        Action 0: up
        Action 1: down
        Action 2: left
        Action 3: right
        '''
        if action == 0:
            if current[0] > 0 and self.grid[current[0] - 1, current[1]] != 'X':
                return np.array([current[0] - 1, current[1]])
        elif action == 1:
            if current[0] < self.grid.shape[0] - 1 and self.grid[current[0] + 1, current[1]] != 'X':
                return np.array([current[0] + 1, current[1]])
        elif action == 2:
            if current[1] > 0 and self.grid[current[0], current[1] - 1] != 'X':
                return np.array([current[0], current[1] - 1])
        elif action == 3:
            if current[1] < self.grid.shape[1] - 1 and self.grid[current[0], current[1] + 1] != 'X':
                return np.array([current[0], current[1] + 1])
        return current

def epsilon_greedy(Q, state, nA, epsilon = 0.2):
    '''
    This function will return the action based on the epsilon greedy policy.
    '''
    if np.random.random() > epsilon: 
        action = np.argmax(Q[state])
    else: 
        action = np.random.choice(np.arange(nA))
    return action
def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.2):
    '''
    This function will generate Q from SARSA algorithm
    '''
    Q = defaultdict(lambda: np.zeros(4))
    for t in range(n_episodes):
        epsilon = 1/(t+1)
        state = env.reset()
        action = epsilon_greedy(Q, state, 4, epsilon)
        done = False
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, 4, epsilon)
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            action = next_action
    return Q
def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.2):
    '''
    This function will generate Q from Q-learning algorithm
    '''
    Q = defaultdict(lambda: np.zeros(4))
    for t in range(n_episodes):
        epsilon = 1/(t+1)
        state = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(Q, state, 4, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
    return Q
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lazy and lame gridworld')
    parser.add_argument('-p', metavar='p', type=float, default=0.2,
                        help='probability of moving in the desired direction')
    parser.add_argument('--random', action='store_true', default=False, help='generate a random grid')
    parser.add_argument('--gridfile', type=argparse.FileType('r'), default=None, help='load a grid from a file')
    parser.add_argument('--r', type=float, default=-0.2, help='reward for each step')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)
    env = GridWorld(args.p, args.r, args.gridfile)
    env.reset()
    env.render()
    while not env.done:
        action = np.random.randint(0, 4)
        state, reward, done, _, _ = env.step(action)
        #display the image with matplotlib, update the image if recalled
        env.render()
    print("-------------------")
    print("Sarsa")
    Q = sarsa(env, 10000)
    env.reset()
    env.render()
    #play the game
    action = epsilon_greedy(Q, state, 4, 0)
    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        next_action = epsilon_greedy(Q, next_state, 4, 0)
        action = next_action
    print("-------------------")
    print("q-learning")
    Q = q_learning(env, 10000)
    env.reset()
    env.render()
    #play the game
    action = epsilon_greedy(Q, state, 4, 0)
    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        next_action = epsilon_greedy(Q, next_state, 4, 0)
        action = next_action
        
    #print the policy as a heatmap
    #plt.imshow(policy, cmap='hot', interpolation='nearest')
    #plt.show()
    #now


