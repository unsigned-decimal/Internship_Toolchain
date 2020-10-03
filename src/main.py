from model import Model
import stormpy.examples
import stormpy.examples.files

def main():
    path1 = "../models/maze.prism"
    path2 = "../models/maze.drn"
    #path3 = "../models/die.jani"
    m = Model(path1) 

    p1 = "P=?[F \"goal\"]"
    p2 = "Rmin=?[F \"goal\"]"
    p3 = "P=?[F \"goal\"]"
    m.parse_properties(p3)
    m.build_model(properties=True)
    m.build_pmc()
    print(m.pmc)
    result = m.model_checking()
    '''
    for state in m.pmc.states:
        for action in state.actions:
            for transition in action.transitions:
                print("State", state, " using action", action," goes to state",\
                         transition.column," with propability", transition.value())
    '''

    initial_state = m.pmc.initial_states[0]
    func = result.at(initial_state)
    cnt = 0
    for x in result.get_values():
        print("value", cnt," =", x.to_smt2())
        cnt += 1
    #print(result.get_values())

    
    #m.print_model_type()
    #m.print_nr_states()
    #m.print_actions()
    #m.print_transitions() 
    
    

if __name__ == "__main__":
    main()