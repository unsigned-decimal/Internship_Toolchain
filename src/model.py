import os
import stormpy

class Model:

    def __init__(self, model_file):
        self.file_type = ""
        self.properties = None
        self.program = self.load_model(model_file)
        self.model = None
        
    def load_model(self, model_file):
        self.properties = None
        self.file_type = os.path.splitext(model_file)[1]
        if self.file_type == ".prism":
            return stormpy.parse_prism_program(model_file)
        else:
            raise Exception("Model file type not supported") 

    def parse_properties(self, properties):
        if self.file_type == ".prism":
            self.properties = stormpy.parse_properties_for_prism_program(properties, self.program, None)
        else:
            raise Exception("Model file type not supported")      

    def build_model(self):
        if self.file_type == ".prism":
            self.model = stormpy.build_model(self.program, self.properties)
        else:
            raise Exception("Model file type not supported") 
    

    # Print functions
    def print_model_type(self):
        print(self.model.model_type)
    
    def print_transitions(self):
        for state in self.model.states:
            for action in state.actions:
                for transition in action.transitions:
                    print("State", state, " using action", action," goes to state",\
                         transition.column," with propability", transition.value())

    

