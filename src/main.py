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
    m.parse_properties(p2)
    m.build_model(properties=True)
    m.build_pmc()
    print(m.pmc)
    result = m.model_checking()
    print(result)
    

    
    #m.print_model_type()
    #m.print_nr_states()
    #m.print_actions()
    #m.print_transitions() 
    
    

if __name__ == "__main__":
    main()