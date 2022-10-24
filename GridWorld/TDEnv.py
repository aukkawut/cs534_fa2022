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
def generate_random_world(size,pw = 0.1, prp = 0.05, prn = 0.05,pwh = 0):
    '''
    This function generates a random gridworld.
    size: size of the gridworld
    pw: probability of a wall in the gridworld
    prp: probability of a positive reward in the gridworld
    prn: probability of a negative reward in the gridworld
    pwh: number of a wormhole in the gridworld
    '''
    grid = np.zeros((size+1, size+1), dtype='object')
    #make sure that outside the grid is a wall
    grid[0, :] = 'X'
    grid[-1, :] = 'X'
    grid[:, 0] = 'X'
    grid[:, -1] = 'X'
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if grid[i,j] != 'W':
                if np.random.random() < pw:
                    grid[i, j] = 'X'
                elif np.random.random() < prp:
                    grid[i, j] = str(np.random.randint(1, 9))
                elif np.random.random() < prn:
                    grid[i, j] = str(-np.random.randint(1, 9))
                else:
                    grid[i, j] = '0'
    grid[size-1, 1] = 'S'
    #place wormhole
    for i in range(pwh):
        while True:
            x = np.random.randint(1, size)
            y = np.random.randint(1, size)
            if grid[x,y] == '0':
                #random alphabet that is not X,A,S
                random_char = chr(np.random.randint(65, 91))
                if random_char != 'X' and random_char != 'A' and random_char != 'S':
                    grid[x,y] = random_char
                    #pick the out point of the wormhole
                    while True:
                        x = np.random.randint(1, size)
                        y = np.random.randint(1, size)
                        if grid[x,y] == '0':
                            grid[x,y] = random_char
                            break                
                break

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
    def __init__(self,p,r,gridFile,pW,prP,prN,pWh,size,maxTimestep = 50) -> None:
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
        self.pW = pW
        self.prP = prP
        self.prN = prN
        self.pWh = pWh
        self.size = size
        self.r = r
        self.read_grid(gridFile)
        self.timestep = 0
        self.maxTimestep = maxTimestep
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
            try:
                self.grid = generate_random_world(self.size,self.pW,self.prP,self.prN,self.pWh)
            except:
                raise Exception('No grid file or size provided')
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
        self.timestep = 0
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
        self.timestep += 1
        self.action = action
        if self.done or self.timestep > self.maxTimestep:
            self.done = True
            return (self.current[0],self.current[1]), self.reward, self.done, self.done,{}
        if np.random.random() < self.p:
            self.envAction = "Normal"
            self.current = self.move(self.current, action)
        elif np.random.random() < 0.5:
            self.envAction = "Double"
            self.current = self.move(self.current, action)
            self.current = self.move(self.current, action)
        else:
            self.envAction = "Backward"
            if action == 0:
                self.current = self.move(self.current, 1)
            elif action == 1:
                self.current = self.move(self.current, 0)
            elif action == 2:
                self.current = self.move(self.current, 3)
            elif action == 3:
                self.current = self.move(self.current, 2)
            #self.current = self.move(self.current, (action + 2)%4)
        #if we hit wormhole, we go to the other wormhole
        #wormhole is represented by a pair of non-numeric character which is not X, A, S
        if str(self.grid[self.current[0], self.current[1]]).isalpha() and self.grid[self.current[0], self.current[1]] != 'X' and self.grid[self.current[0], self.current[1]] != 'A' and self.grid[self.current[0], self.current[1]] != 'S' and not (self.grid[self.current[0], self.current[1]].lstrip("-").isdigit()):
            self.current = np.argwhere(self.grid == self.grid[self.current[0], self.current[1]])[1]
        if str(self.grid[self.current[0], self.current[1]]) != '0' and str(self.grid[self.current[0], self.current[1]]) !='S' and str(self.grid[self.current[0], self.current[1]]).lstrip("-").isdigit():
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
        return (self.current[0], self.current[1]), self.reward, self.done, self.done,{}
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
                elif grid[i, j] == '0' or grid[i, j] == 0:
                    #print empty squares as nothing
                    print(' ', end='\t')
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
    def gridsize(self) -> tuple:
        return (self.grid.shape[0], self.grid.shape[1])

def epsilon_greedy(Q, state, nA, epsilon = 0.2):
    '''
    This function will return the action based on the epsilon greedy policy.
    '''
    if np.random.random() > epsilon: 
        action = np.argmax(Q[state])
    else: 
        action = np.random.choice(np.arange(nA))
    return action
def player_game(env):
    '''
    Let human play the game. Using the arrow key to move the agent.
    '''
    env.reset()
    env.render()
    while True:
        action = input("Enter the action (w,a,s,d): ")
        if action == 'w':
            action = 0
        elif action == 's':
            action = 1
        elif action == 'a':
            action = 2
        elif action == 'd':
            action = 3
        else:
            print("Invalid action")
            continue
        _, reward, done, _,_ = env.step(action)
        env.render()
        if done:
            print(f"Game over. Total reward: {reward}")
            break
def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.9):
    '''
    This function will generate Q from SARSA algorithm
    '''
    Q = defaultdict(lambda: np.zeros(4))
    for t in range(n_episodes):
       # epsilon = 1/(t+1)
        epsilon = max(epsilon - ((epsilon - 0.01)/n_episodes),0.01)
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
def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.9):
    '''
    This function will generate Q from Q-learning algorithm
    '''
    Q = defaultdict(lambda: np.zeros(4))
    for t in range(n_episodes):
        epsilon = max(epsilon - ((epsilon - 0.01)/n_episodes),0.01)
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
def printPolicy(Q, grid_size, grid):
    '''
    This function will print the action that maximize Q as the arrow
    '''
    #raise NotImplementedError
    policy = -1 * np.ones(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            try:
                policy[i,j] = np.argmax(Q[(i,j)])
                if np.sum(Q[(i,j)]) == 0:
                    policy[i,j] = -1
            except:
                pass
    #mask the policy with the grid
    policy = np.ma.masked_where(grid == 'X', policy)
    #print the grid with empty squares replace with our policy
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if not str(grid[i, j]).lstrip("-").isdigit():
                #color X with red
                if grid[i,j] == 'X':
                    print('\033[31m' + grid[i, j] + '\033[0m', end='\t')
                else:
                    #blue
                    print('\033[34m' + grid[i, j] + '\033[0m', end='\t')
            elif policy[i, j] == 0:
                print('↑', end='\t')
            elif policy[i, j] == 1:
                print('↓', end='\t')
            elif policy[i, j] == 2:
                print('←', end='\t')
            elif policy[i, j] == 3:
                print('→', end='\t')
            else:
                if str(grid[i, j]) != '0':
                    #print the reward in green
                    #print(f'{grid[i, j]}', end='\t')
                    print('\033[32m' + str(grid[i, j]) + '\033[0m', end='\t')
                else:
                    print(' ', end='\t')
        print()


if __name__ == '__main__':
    #create colored text for title
    x = """
        \033[1;33m GridWorld\033[0m :
        The lame and silly game
        """
    parser = argparse.ArgumentParser(description=x)
    #make the mode: human, sarsa, and Q
    parser.add_argument('mode', type=str, help='Mode of simulation: human, random, sarsa, q')
    parser.add_argument('-p', metavar='p', type=float, default=0.2,
                        help='probability of moving in the desired direction (Default is 0.2)')
    parser.add_argument('--gridfile', type=argparse.FileType('r'), default=None, help='load a grid from a file')
    parser.add_argument('--r', type=float, default=-0.2, help='reward for each step (Default is -0.2)')
    parser.add_argument('--seed', type=int, default=2, help='random seed (Default is 2)')
    parser.add_argument('--size', type=int, default=5, help='size of the grid (Default is 5)')
    parser.add_argument('-pW', metavar='pW', type=float, default=0.2, help='probability of a wall in a random grid (Default is 0.2)')
    parser.add_argument('-prP', metavar='prP', type=float, default=0.1, help='probability of a positive reward in a random grid (Default is 0.1)')
    parser.add_argument('-prN', metavar='prN', type=float, default=0.1, help='probability of a negative reward in a random grid (Default is 0.1)')
    parser.add_argument('-nWh', metavar='nWh', type=int, default=0, help='number of a wormhole in a random grid (Default is 0)')
    parser.add_argument('-nEp', metavar='nEp', type=int, default=10000, help='number of episodes (Default is 10000)')
    args = parser.parse_args()
    np.random.seed(args.seed)
    env = GridWorld(args.p, args.r, args.gridfile, args.pW, args.prP, args.prN, args.nWh,args.size)
    if args.mode == 'human':
        env.reset()
        env.render()
        player_game(env)
    elif args.mode == 'random':
        env.reset()
        env.render()
        while not env.done:
            action = np.random.randint(0, 4)
            state, reward, done, _, _ = env.step(action)
            env.render()
    elif args.mode == 'sarsa':
        Q = sarsa(env, args.nEp)
        state = env.reset()
        env.render()
        #play the game
        action = epsilon_greedy(Q, state, 4, 0)
        done = False
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            env.render()
            next_action = epsilon_greedy(Q, next_state, 4, 0)
            action = next_action
        #for each point in the grid, print the q value
        printPolicy(Q, env.gridsize(), env.grid)
    elif args.mode == 'q':
        Q = q_learning(env, args.nEp)
        state = env.reset()
        env.render()
        #play the game
        action = epsilon_greedy(Q, state, 4, 0)
        done = False
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            env.render()
            next_action = epsilon_greedy(Q, next_state, 4, 0)
            action = next_action
        printPolicy(Q, env.gridsize(), env.grid)
    else:
        raise Exception("Invalid mode provided.")
    #print the policy as a heatmap
    #plt.imshow(policy, cmap='hot', interpolation='nearest')
    #plt.show()
    #now
    


