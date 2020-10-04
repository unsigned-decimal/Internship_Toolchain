from pomdp import POMDP
from pmc import PMC
import stormpy.examples
import stormpy.examples.files

def main():
    path1 = "../models/maze.prism"
    path2 = "../models/maze.drn"
    #path3 = "../models/die.jani"
    pomdp = POMDP(path1) 

    p1 = "P=?[F \"goal\"]"
    p2 = "Rmin=?[F \"goal\"]"
    p3 = "P=?[F \"goal\"]"

    pomdp.parse_properties(p1)
    pomdp.build_model()

    pomdp.print_model_type()
    pomdp.print_nr_states()
    pomdp.print_actions()
    pomdp.print_transitions() 
    print(pomdp.model)

    pmc = PMC(pomdp.build_pmc())
    pmc.print_model_type()
    pmc.print_nr_states()
    pmc.print_actions()
    pmc.print_transitions()
    print(pmc.model)
    pmc.model_checking(pomdp.properties[0])
    pmc.print_results()

if __name__ == "__main__":
    main()