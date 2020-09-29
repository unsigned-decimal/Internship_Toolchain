import os
import stormpy
import stormpy.pomdp

class Model:

    def __init__(self, model_file):
        self.file_type = ""
        self.properties = None
        self.program = self.load_model(model_file)
        self.model = None
        self.pmc = None
        
    def load_model(self, model_file):
        self.properties = None
        self.file_type = os.path.splitext(model_file)[1]
        if self.file_type == ".prism":
            return stormpy.parse_prism_program(model_file)
        elif self.file_type == ".drn":
            return model_file
        elif self.file_type == ".jani":
            jani_program, self.properties = stormpy.parse_jani_model(model_file)
            return jani_program
        else:
            raise Exception("Model file type not supported") 

    def parse_properties(self, properties):
        if self.file_type == ".prism":
            self.properties = stormpy.parse_properties_for_prism_program(properties, self.program)
        elif self.file_type == ".jani":
            self.properties = stormpy.parse_properties_for_jani_model(properties, self.program)
        elif self.file_type == ".drn":
            self.properties = stormpy.parse_properties_without_context(properties)
        else:
            raise Exception("Model file type not supported")      

    def build_model(self, properties=False):
        if self.file_type == ".prism" or self.file_type == ".jani":
            self.model = stormpy.build_model(self.program, self.properties if properties else None)
        elif self.file_type == ".drn":
            self.model = stormpy.build_model_from_drn(self.program)
        else:
            raise Exception("Model file type not supported") 

    def build_pmc(self):
        #self.model = stormpy.pomdp.make_canonic(self.model)
        self.model = stormpy.pomdp.make_simple(self.model)
        memoryBuilder = stormpy.pomdp.PomdpMemoryBuilder()

        memory = memoryBuilder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 2)
        self.model = stormpy.pomdp.unfold_memory(self.model, memory)
        self.pmc = stormpy.pomdp.apply_unknown_fsc \
            (self.model, stormpy.pomdp.PomdpFscApplicationMode.simple_linear)

    def model_checking(self):
        return stormpy.model_checking(self.pmc, self.properties[0])

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

    

