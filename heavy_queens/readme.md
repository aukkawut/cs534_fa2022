# Heavy queen

It is just a n-queen but each queen has its own tendency to move (i.e. the queen with high weight will not move but the queen with low weight will move).
This is penalized by the cost calculated by distance moved multiply by its weight squared.

We explored the A* algorithm, greedy search algorithm, hill climbing, and simulated annealing algorithm here. The one being uploaded here is the one I implemented.

Also note that hill climbing is a greedy algorithm so the tendency to move is ignored.
