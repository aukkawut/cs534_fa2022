import numpy as np
import gym
import argparse
from collections import defaultdict
import os
from pathlib import Path
#import sys
import warnings
os.system("")
#import matplotlib.pyplot as plt
import os
def is_valid_file(parser, arg):
    if Path(arg).is_file():
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(Path(arg).resolve(), 'r')  # return an open file handle
def generate_random_world(size,pw = 0.1, prp = 1, prn = 1,pwh = 0):
    '''
    This function generates a random gridworld.
    size: size of the gridworld
    pw: probability of a wall in the gridworld
    prp: number of a positive reward in the gridworld
    prn: number of a negative reward in the gridworld
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
        #place positive reward
    for i in range(prp):
        while True:
            x = np.random.randint(1, size)
            y = np.random.randint(1, size)
            if grid[x,y] == '0':
                #random int between 1 - 9
                random_int = np.random.randint(1, 10)
                grid[x,y] = str(random_int)
                break
        #place negative reward
    for i in range(prn):
        while True:
            x = np.random.randint(1, size)
            y = np.random.randint(1, size)
            if str(grid[x,y]) == '0' :
                #random int between 1 - 9
                random_int = np.random.randint(1, 10)
                grid[x,y] = str(-1*random_int)
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
        self.truncated = False
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
            #if the first entry is not wall, then add a wall enclosing the grid
            if self.grid[0,0] != 'X':
                self.grid = np.insert(self.grid, 0, 'X', axis=0)
                self.grid = np.insert(self.grid, 0, 'X', axis=1)
                self.grid = np.insert(self.grid, self.grid.shape[0], 'X', axis=0)
                self.grid = np.insert(self.grid, self.grid.shape[1], 'X', axis=1)
        except:
            try:
                self.grid = generate_random_world(self.size,self.pW,self.prP,self.prN,self.pWh)
            except:
                raise Exception('No grid file or size provided')
        self.observation_space = gym.spaces.Discrete(self.grid.shape[0] * self.grid.shape[1])
        self.start = np.argwhere(self.grid == 'S')[0]
        self.current = self.start
        #print("start: ", self.start)
    def reset(self) -> None:
        '''
        This function resets the environment.
        '''
        self.current = self.start
        self.reward = 0
        self.done = False
        self.action = None
        self.envAction = None
        self.truncated = False
        self.timestep = 0
        return (self.current[0],self.current[1])
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
        #if self.truncated:
        #    raise Exception('The episode is done. Please reset the environment.')
        self.timestep += 1
        self.action = action
        if self.done or self.timestep > self.maxTimestep:
            if self.timestep > self.maxTimestep:
                self.truncated = True
                self.done = False
            return (self.current[0],self.current[1]), self.reward, self.done, self.truncated,{}
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
        return (self.current[0], self.current[1]), self.reward, self.done, self.truncated,{}
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
def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.9, end_epsilon=0.1,decay = True):
    '''
    This function will generate Q from SARSA algorithm
    '''
    Q = defaultdict(lambda: np.zeros(4))
    SP = np.zeros(env.gridsize())
    SPT = np.zeros(env.gridsize())
    total_rewards = []
    denom = 0
    for t in range(n_episodes):
       # epsilon = 1/(t+1)
        if decay:
            epsilon = max(epsilon - ((epsilon - end_epsilon)/(0.5*n_episodes)),end_epsilon)
        state = env.reset()
        SP[state] += 1
        action = epsilon_greedy(Q, state, 4, epsilon)
        done = False
        truncated = False
        reward_e = 0
        eplen = 0
        while not done:
            if truncated:
                break
            eplen += 1
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, 4, epsilon)
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            SP[state] += 1
            action = next_action
            reward_e += reward
        if done:
            denom +=1
        total_rewards.append(reward_e)
        SPT = SPT + SP/np.sum(SP)
        SP = np.zeros(env.gridsize())
    SPT = SPT/n_episodes
    if denom == 0:
        warnings.warn(f"Agent never reached the terminal state (halting condition: step > {env.maxTimestep}) Changing halting by setting -maxT to higher value", RuntimeWarning, stacklevel=2)
            #denom = np.finfo(float).eps
        return Q, np.array(total_rewards), SPT, -np.inf
    else:
        return Q, np.array(total_rewards), SPT, np.sum(total_rewards)/denom
def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.9, end_epsilon=0.01,decay = True):
    '''
    This function will generate Q from Q-learning algorithm
    '''
    Q = defaultdict(lambda: np.zeros(4))
    total_reward = []
    SP = np.zeros(env.gridsize())
    SPT = np.zeros(env.gridsize())
    denom = 0
    for t in range(n_episodes):
        if decay:
            epsilon = max(epsilon - ((epsilon - end_epsilon)/(0.5*n_episodes)),end_epsilon)
        state = env.reset()
        done = False
        reward_e = 0
        #print(state)
        SP[state] += 1
        eplen = 0
        truncated = False
        while not done:
            if truncated:
                break
            eplen +=1
            action = epsilon_greedy(Q, state, 4, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            #print(state)
            SP[state] += 1
            reward_e += reward
        if done:
            denom +=1
        SP = SP/np.sum(SP)
        total_reward.append(reward_e)
        SPT = SPT + SP
        SP = np.zeros(env.gridsize())
    SPT = SPT/n_episodes
    #return Q, np.array(total_rewards), SPT
    if denom == 0:
        warnings.warn(f"Agent never reached the terminal state (halting condition: step > {env.maxTimestep}) Changing halting by setting -maxT to higher value", RuntimeWarning, stacklevel=2)
            #denom = np.finfo(float).eps
        return Q, np.array(total_reward), SPT, -np.inf
    else:
        return Q, np.array(total_reward), SPT, np.sum(total_reward)/denom
def printPolicy(Q, grid_size, grid):
    '''
    This function will print the action that maximize Q as the arrow
    '''
    #raise NotImplementedError
    print("Policy")
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
            if not str(grid[i, j]).lstrip("-").isdigit() and str(grid[i,j]) != 'S':
                #color X with red
                if grid[i,j] == 'X':
                    print('\033[31m' + grid[i, j] + '\033[0m', end='\t')
                else:
                    #blue
                    print('\033[34m' + grid[i, j] + '\033[0m', end='\t')
            elif policy[i, j] == 0:
                #if windows, use '^' instead of '↑'
                if os.name == 'nt':
                    print('^', end='\t')
                else:
                    print('↑', end='\t')
            elif policy[i, j] == 1:
                if os.name == 'nt':
                    print('v', end='\t')
                else:
                    print('↓', end='\t')
            elif policy[i, j] == 2:
                if os.name == 'nt':
                    print('<', end='\t')
                else:
                    print('←', end='\t')
            elif policy[i, j] == 3:
                if os.name == 'nt':
                    print('>', end='\t')
                else:
                    print('→', end='\t')
            else:
                if str(grid[i, j]) != '0' and str(grid[i, j]) != 'S':
                    #print the reward in green
                    #print(f'{grid[i, j]}', end='\t')
                    print('\033[32m' + str(grid[i, j]) + '\033[0m', end='\t')
                else:
                    print(' ', end='\t')
        print()
def Heatmap(SP,grid):
    '''
    This function will print the percentage (in int) of the state visitation masked by the grid
    '''
    print("Heatmap of state visitation (yellow means more than average)")
    SP = SP*100
    SP = np.ma.masked_where(grid == 'X', SP)
    for i in range(SP.shape[0]):
        for j in range(SP.shape[1]):
            if not str(grid[i, j]).lstrip("-").isdigit() and str(grid[i,j]) != 'S':
                #color X with red
                if grid[i,j] == 'X':
                    print('\033[31m' + grid[i, j] + '\033[0m', end='\t')
                else:
                    #blue
                    print('\033[34m' + grid[i, j] + '\033[0m', end='\t')
            else:
                if str(grid[i, j]) != '0' and str(grid[i, j]) != 'S':
                    #print the reward in green
                    #print(f'{grid[i, j]}', end='\t')
                    print('\033[32m' + str(grid[i, j]) + '\033[0m', end='\t')
                else:
                    if SP[i,j] == 0:
                        print(' ', end='\t')
                    else:
                        #color the state that visited more than average with yellow
                        if SP[i,j] > np.mean(SP):
                            print('\033[33m' + f'{SP[i,j]:.1f}' + '\033[0m', end='\t')
                        else:
                            print(f'{SP[i,j]:.1f}', end='\t')
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
    parser.add_argument('-p', metavar='p', type=float, default=0.7,
                        help='probability of moving in the desired direction (Default is 0.7)')
    parser.add_argument('--gridfile', type=argparse.FileType('r'), default=None, help='load a grid from a file')
    parser.add_argument('--r', type=float, default=-0.2, help='reward for each step (Default is -0.2)')
    parser.add_argument('--seed', type=int, default=2, help='random seed (Default is 2)')
    parser.add_argument('--size', type=int, default=5, help='size of the grid (Default is 5)')
    parser.add_argument('-pW', metavar='pW', type=float, default=0.2, help='probability of a wall in a random grid (Default is 0.2)')
    parser.add_argument('-nrP', metavar='nrP', type=int, default=1, help='number of a positive reward in a random grid (Default is 1)')
    parser.add_argument('-nrN', metavar='nrN', type=int, default=1, help='number of a negative reward in a random grid (Default is 1)')
    parser.add_argument('-nWh', metavar='nWh', type=int, default=0, help='number of a wormhole in a random grid (Default is 0)')
    parser.add_argument('-maxT', metavar='maxT', type=int, default=100, help='maximum number of steps in an episode (Default is 100)')
    parser.add_argument('-start_eps', metavar='start_eps', type=float, default=0.9, help='starting epsilon for epsilon-greedy (Default is 0.9)')
    parser.add_argument('-end_eps', metavar='end_eps', type=float, default=0.01, help='ending epsilon for epsilon-greedy (Default is 0.01)')
    parser.add_argument('-alpha', metavar='alpha', type=float, default=0.5, help='learning rate (Default is 0.5)')
    parser.add_argument('-gamma', metavar='gamma', type=float, default=0.9, help='discount factor (Default is 0.9)')
    parser.add_argument('-nEp', metavar='nEp', type=int, default=10000, help='number of episodes (Default is 10000)')
    parser.add_argument('-nEpT', metavar='nEpT', type=int, default=100, help='number of episodes for testing (Default is 100)')
    parser.add_argument('-epsDecay', metavar='epsDecay', type=bool, default=True, help='epsilon decay as [(eps - end_eps)/0.5n] (Default is True)')
    parser.add_argument('-plotext', metavar='plotext', type=bool, default=False, help='plot the graph with plotext (Default is False, will use matplotlib)')
    args = parser.parse_args()
    if args.plotext:
        import plotext as plt
    else:
        import matplotlib.pyplot as plt
    np.random.seed(args.seed)
    env = GridWorld(args.p, args.r, args.gridfile, args.pW, args.nrP, args.nrN, args.nWh,args.size, args.maxT)
    if args.mode == 'human':
        env.reset()
        env.render()
        player_game(env)
    elif args.mode == 'random':
        env.reset()
        env.render()
        while not env.done:
            if env.truncated:
                warnings.warn("Episode truncated at {} steps".format(env.maxTimestep), RuntimeWarning, stacklevel=2)
                break
            action = np.random.randint(0, 4)
            state, reward, done, truncated, _ = env.step(action)
            env.render()
    elif args.mode == 'sarsa':
        Q, r,SP,mr = sarsa(env, args.nEp, epsilon = args.start_eps, end_epsilon = args.end_eps, alpha = args.alpha, gamma = args.gamma, decay = args.epsDecay)
        state = env.reset()
        env.render()
        #play the game
        action = epsilon_greedy(Q, state, 4, 0)
        done = False
        truncated = False
        while not done:
            if truncated:
                break
            next_state, reward, done, truncated, _ = env.step(action)
            env.render()
            next_action = epsilon_greedy(Q, next_state, 4, 0)
            action = next_action
        #for each point in the grid, print the q value
        printPolicy(Q, env.gridsize(), env.grid)
        print('Average training reward: ', mr)
        Heatmap(SP,env.grid)
        #plot average 10 training reward
        #set figure size
        if not args.plotext:
            plt.figure(figsize=(10, 5))
        else:
            plt.plotsize(100, 20)
        plt.plot(np.convolve(r, np.ones((30,))/30, mode='valid'))
        plt.title('Average 30 training reward of Sarsa')
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
        #show other parameters below the plot
        #plt.annotate('p = ' + str(args.p) + ', r = ' + str(args.r) + ', alpha = ' + str(args.alpha) + ', gamma = ' + str(args.gamma) + ', epsilon = ' + str(args.start_eps) + ', end_epsilon = ' + str(args.end_eps), (0,0), (20, -40), xycoords='axes fraction', textcoords='offset points', va='top')
        #plt.figtext(0.5, 0.01, 'p = ' + str(args.p) + ', r = ' + str(args.r) + ', alpha = ' + str(args.alpha) + ', gamma = ' + str(args.gamma) + ', epsilon = ' + str(args.start_eps) + ', end_epsilon = ' + str(args.end_eps), wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()
        r = []
        denom = 0
        for i in range(args.nEpT):
            state = env.reset()
            done = False
            truncated = False
            while not done:
                if truncated:
                    break
                action = epsilon_greedy(Q, state, 4, 0)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
            if done:
                denom +=1
            r.append(reward)
        if denom == 0:
            warnings.warn(f"Agent never reached the terminal state (halting condition: step > {env.maxTimestep}) Changing halting by setting -maxT to higher value", RuntimeWarning, stacklevel=2)
            #denom = np.finfo(float).eps
            print(f'Average testing reward: {-np.inf} ({np.mean(r)})')
        else:
            print('Average testing reward: ', np.sum(r)/denom)
    elif args.mode == 'q':
        Q,r,SP,mr = q_learning(env, args.nEp, epsilon = args.start_eps, end_epsilon = args.end_eps, alpha = args.alpha, gamma = args.gamma, decay = args.epsDecay)
        state = env.reset()
        env.render()
        #play the game
        action = epsilon_greedy(Q, state, 4, 0)
        done = False
        truncated = False
        while not done:
            if truncated:
                break
            next_state, reward, done, truncated, _ = env.step(action)
            env.render()
            next_action = epsilon_greedy(Q, next_state, 4, 0)
            action = next_action
        printPolicy(Q, env.gridsize(), env.grid)
        Heatmap(SP,env.grid)
        print('Average training reward: ', mr)
        #plot average 10 training reward
        if not args.plotext:
            plt.figure(figsize=(10, 5))
        else:
            plt.plotsize(100, 20)
        #plt.figure(figsize=(10, 5))
        plt.plot(np.convolve(r, np.ones((30,))/30, mode='valid'))
        plt.title('Average 30 training reward of Q-learning')
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
        #plt.annotate('p = ' + str(args.p) + ', r = ' + str(args.r) + ', alpha = ' + str(args.alpha) + ', gamma = ' + str(args.gamma) + ', epsilon = ' + str(args.start_eps) + ', end_epsilon = ' + str(args.end_eps), (0,0), (50, 40), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.show()
        #testing: run the game for nEpT episodes and print the average reward
        r = []
        denom = 0
        for i in range(args.nEpT):
            state = env.reset()
            done = False
            truncated = False
            while not done:
                if truncated:
                    break
                action = epsilon_greedy(Q, state, 4, 0)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
            if done:
                denom += 1
            r.append(reward)
        if denom == 0:
            warnings.warn(f"Agent never reached the terminal state (halting condition: step > {env.maxTimestep}) Changing halting by setting -maxT to higher value", RuntimeWarning, stacklevel=2)
            print(f'Average testing reward: {-np.inf} ({np.mean(r)})')
            #denom = np.finfo(float).eps
        else:
            print('Average testing reward: ', np.sum(r)/denom)
        
    else:
        raise Exception("Invalid mode provided.")
#python TDEnv.py --size 10 --seed 213215 --r -0.012 -alpha 0.6 -gamma 0.9 -nrP 1 -nrN 1 -p 1 -pW 0.2 -start_eps 1 -end_eps 0 -nEp 1000 -nWh 1 -epsDecay True q