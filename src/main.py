from pomdp import POMDP
from pmc import PMC

import stormpy.examples
import stormpy.examples.files
import stormpy.pars
from pycarl.cln.cln import Rational
import time
def main():
    path1 = "../models/maze.prism"
    path2 = "../models/maze.drn"
    #path3 = "../models/die.jani"
    path4 = "../models/network2.prism"
    pomdp = POMDP()
    pomdp.load_model(path1) 

    p1 = "R=?[F \"goal\"]"
    p2 = "R<1[F \"goal\"]"
    p3 = "P>=0.7[X s=0]"
    p4 = "R{\"dropped_packets\"}min=?[F sched=0 & t=T-1 & k=K-1 ]"

    pomdp.parse_properties(p1)
    pomdp.build_model()
    print(pomdp.model)

    pmc = PMC(pomdp.build_pmc())
    print(pmc.model)
    pmc.set_parameters([0.4 for _ in range(pmc.nr_parameters)])
    pmc.model_checking(pomdp.properties[0])
    #pmc.print_results()

    print("***** Program End *****")

if __name__ == "__main__":
    main()