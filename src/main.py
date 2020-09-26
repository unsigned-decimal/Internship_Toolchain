from model import Model

def main():
    m = Model("../models/maze.prism") 
    p = "Rmin=?[F s=10 ]"
    m.parse_properties(p)  
    m.build_model()
    m.print_model_type()
    m.print_transitions() 
    print("nr_states:", m.model.nr_states)
    print("nr_observations:", m.model.nr_observations)
    

if __name__ == "__main__":
    main()