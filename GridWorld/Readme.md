# Gridworld

Yes, this is RL stuff. Ripoff frozen lake

## How to run this?

### Requirements
Make sure that you have `gym`, `matplotlib` and `numpy` installed.

### Getting Start
Start from the helping function
```
python TDEnv.py --help
```
### Testing
If you have your grid file (tab delimited text file) where 0 means empty, [-9,9] \ {0} means reward, S means start, and X means wall. Then, you can run 
```
python TDEnv.py --gridfile <your_file> <human|random|sarsa|q>
```
for running this.

### Randomize your board
If you want to create a random board, you can customize the board with the following tags and commands
```
python TDEnv.py --seed <seed> --size <size> -pW <wall_prob> -nrP <num_pos_reward> -nrN <num_neg_reward> -nWh <num_wormhole_pairs> <human|random|sarsa|q>
```
You can set the random seed, the size of the board, the probability of having a wall in any spot, how many positive reward, how many negative reward, and how many wormholes pairs.

### Training Customization
There are some customization on training, this includes the parameters like alpha and gamma, the epsilon decay, number of runs for finding Q, number of moves to terminated, etc. Please use the helping function for more details.
