import os
import stormpy
import stormpy.pomdp

class POMDP:

    def __init__(self):
        self.file_type = ""
        self.file_path = ""
        self.properties = None
        self.program = None
        self.model = None
        
    def load_model(self, model_file):
        self.properties = None
        self.file_type = os.path.splitext(model_file)[1]
        self.file_path = model_file
        if self.file_type == ".prism":
            self.program = stormpy.parse_prism_program\
                (model_file, prism_compat=True, simplify=False)
        elif self.file_type == ".drn":
            self.program = None
        elif self.file_type == ".jani":
            self.program, self.properties = stormpy.parse_jani_model(model_file)
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
    

    def build_model(self):
        if self.file_type == ".prism" or self.file_type == ".jani":
            self.model = stormpy.build_model(self.program, self.properties)
        elif self.file_type == ".drn":
            parser_options = stormpy.DirectEncodingParserOptions()
            parser_options.build_choice_labels = True
            self.model = stormpy.build_model_from_drn(self.file_path, parser_options)
        else:
            raise Exception("Model file type not supported") 

    def build_pmc(self):
        self.model = stormpy.pomdp.make_canonic(self.model)
        self.model = stormpy.pomdp.make_simple(self.model)
        memoryBuilder = stormpy.pomdp.PomdpMemoryBuilder()
        memory = memoryBuilder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 2)
        self.model = stormpy.pomdp.unfold_memory(self.model, memory)

        return stormpy.pomdp.apply_unknown_fsc \
            (self.model, stormpy.pomdp.PomdpFscApplicationMode.simple_linear)

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

    

