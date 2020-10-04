import os
import stormpy
import stormpy.pomdp

class PMC:

    def __init__(self, model):
        self.model = model  
        self.result = None  

    def model_checking(self, property):
        self.result = stormpy.model_checking(self.model, property)

    # Print functions
    def print_model_type(self):
        print("Model type:",self.model.model_type)

    def print_nr_states(self):
        print("Number of states:", self.model.nr_states)
    
    def print_actions(self):
        for state in self.model.states:
            print("State", state, " has actions", [str(action) for action in state.actions])

    def print_transitions(self):
        for state in self.model.states:
            for action in state.actions:
                for transition in action.transitions:
                    print("State", state, " using action", action," goes to state",\
                            transition.column," with propability", transition.value())
    
    def print_results(self):
        cnt = 0
        for x in self.result.get_values():
            print("value", cnt," =", x.to_smt2())
            cnt += 1
        #print(self.result.get_values())

    

