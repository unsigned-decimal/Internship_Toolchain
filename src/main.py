from pomdp_solver import *
import stormpy.examples
import stormpy.examples.files
import stormpy.pars
from pycarl.cln.cln import Rational
import time

def prism_maze():
    path = "../models/maze.prism"
    prop = "R=?[F \"goal\"]"
    prop ="Rmin=?[F \"goal\" ]"
    solve_pomdp(path, prop)

def prism_maze2():
    path = "../models/maze2.prism"
    prop = "Rmin=?[F \"target\"]"
    solve_pomdp(path, prop)

def drn_maze():
    print("***** Starting main_drn *****")
    path = "../models/maze.drn"
    prop = "R=?[F \"goal\"]"
    action_reward = [1.0 for _ in range(54)]
    solve_pomdp(path, prop)

def prism_network2():
    print("***** Starting main_prism *****")
    path = "../models/network2.prism"
    prop = "R{\"dropped_packets\"}min=?[F sched=0 & t=T-1 & k=K-1 ]"
    solve_pomdp(path, prop, undefined_constants="K=5, T=2")

def prism_3x3grid():
    path = "../models/3x3grid.prism"
    prop = "Rmin=? [ F \"goal\" ]"
    solve_pomdp(path, prop)

def prism_4x4grid():
    path = "../models/4x4grid.prism"
    prop = "Rmin=? [ F \"goal\" ]"
    solve_pomdp(path, prop)

def prism_crypt3():
    path = "../models/crypt3.prism"
    prop = "P=? [ F \"goal\" ]"
    solve_pomdp(path, prop)

def prism_crypt5():
    path = "../models/crypt5.prism"
    prop = "P=? [ F \"goal\" ]"
    solve_pomdp(path, prop)

def prism_repudiation():
    path = "../models/repudiation_pomdp.prism"
    prop = "Pmax=? [ F \"unfair\" ]"
    solve_pomdp(path, prop, undefined_constants="K=4")

if __name__ == "__main__":
    #Division by zero error
    #prism_maze()
    #drn_maze()
    #prism_4x4grid()
    #prism_network2()
    prism_maze2()

    #Other errors
    #prism_crypt3()
    #prism_crypt5()

    #prism_3x3grid()
    #prism_repudiation()
    


    