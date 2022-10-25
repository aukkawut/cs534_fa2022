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
