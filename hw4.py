from rl import *
from mdp import *

#Returns the grid with the given cost.
def grid_MDP(cost):
    return GridMDP([[cost, cost, cost, cost],
                    [cost, None, None, +1],
                    [cost, cost, cost, -1],
                    [cost, cost, cost, cost]],
                    terminals=[(3, 2), (3, 1)],
                    gamma=0.99)

north = (0, 1)
south = (0,-1)
west = (-1, 0)
east = (1, 0)

#Embedding the policy from 2c.
policy = {
    (0, 3):  east, (1, 3): east,  (2, 3): east,   (3, 3): south,
    (0, 2): north,                                (3, 2): None,
    (0, 1): north, (1, 1): west,  (2, 1): west,   (3, 1): None,
    (0, 0): north, (1, 0): west,  (2, 0): west,   (3, 0): west, 
}


for learning_cost in [-0.1, -0.08, -0.04, -0.02, -0.001]:
    utilities = []
    sequential_decision_environment = grid_MDP(learning_cost)
    learned_agent = PassiveTDAgent(policy, sequential_decision_environment, lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(learned_agent,sequential_decision_environment)
    print("Learning cost = ", learning_cost)
    utilities.append(learned_agent.U)
    print("Utilities: ",utilities)
