from pomdp import POMDP
from pmc import PMC

import stormpy.examples
import stormpy.examples.files
import stormpy.pars
from pycarl.cln.cln import Rational
import time

def main_maze_prism():
    print("***** Starting main_prism *****")
    path = "../models/maze.prism"
    prop = "R=?[F \"goal\"]"
    pomdp = POMDP()
    pomdp.load_model(path) 

    pomdp.parse_properties(prop)
    if pomdp.has_undefined_constants():
        pomdp.set_undefined_constants("K=5, T=2")
    pomdp.build_model()
    print(pomdp.model)

    pmc = PMC(pomdp.build_pmc())
    print(pmc.model)

    pmc.instantiate_parameters([0.4 for _ in range(pmc.nr_parameters)])
    pmc.model_checking(pomdp.properties[0])
    pmc.print_results()

    print("***** Program End *****")

def main_maze_drn():
    print("***** Starting main_drn *****")
    path = "../models/maze.drn"
    prop = "R=?[F \"goal\"]"

    pomdp = POMDP()
    pomdp.load_model(path) 

    pomdp.parse_properties(prop)
    pomdp.build_model()

    #maze.drn has no reward model, so we need to create one.
    reward_models = {}
    action_reward = [1.0 for state in pomdp.model.states for action in state.actions]
    action_reward[0] = 0.0
    action_reward[-1] = 0.0 
    reward_models[''] = stormpy.SparseRewardModel(optional_state_action_reward_vector=action_reward)

    components = stormpy.SparseModelComponents(transition_matrix=pomdp.model.transition_matrix\
        , state_labeling=pomdp.model.labeling, reward_models=reward_models)
    components.choice_labeling = pomdp.model.choice_labeling
    components.observability_classes = pomdp.model.observations
    new_pomdp = stormpy.storage.SparsePomdp(components)
    pomdp.model = new_pomdp
    print(pomdp.model)

    pmc = PMC(pomdp.build_pmc())
    print(pmc.model)

    pmc.instantiate_parameters([0.4 for _ in range(pmc.nr_parameters)])
    pmc.model_checking(pomdp.properties[0])
    pmc.print_results()

    print("***** Program End *****")

def main_network_prism():
    print("***** Starting main_prism *****")
    path = "../models/network2.prism"
    prop = "R{\"dropped_packets\"}min=?[F sched=0 & t=T-1 & k=K-1 ]"

    pomdp = POMDP()
    pomdp.load_model(path)

    pomdp.parse_properties(prop)
    if pomdp.has_undefined_constants():
        pomdp.set_undefined_constants("K=5, T=2")
    pomdp.build_model()
    print(pomdp.model)

    pmc = PMC(pomdp.build_pmc())
    print(pmc.model)

    pmc.instantiate_parameters([0.4 for _ in range(pmc.nr_parameters)])
    pmc.model_checking(pomdp.properties[0])
    pmc.print_results()

    print("***** Program End *****")

if __name__ == "__main__":
    #main_maze_prism()
    #main_maze_drn()
    main_network_prism()