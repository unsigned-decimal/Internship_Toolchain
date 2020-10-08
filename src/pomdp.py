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

    def build_pmc(self,\
        pomdp_memory_pattern=stormpy.pomdp.PomdpMemoryPattern.selective_counter,\
        nr_memory_state=2,\
        fsc_application_mode=stormpy.pomdp.PomdpFscApplicationMode.simple_linear):
        
        # Checking model to determine if it incomplete.  
        self.check_model()

        self.model = stormpy.pomdp.make_canonic(self.model)
        self.model = stormpy.pomdp.make_simple(self.model)
        memoryBuilder = stormpy.pomdp.PomdpMemoryBuilder()
        memory = memoryBuilder.build(pomdp_memory_pattern, nr_memory_state)
        self.model = stormpy.pomdp.unfold_memory(self.model, memory)

        return stormpy.pomdp.apply_unknown_fsc(self.model, fsc_application_mode)

    def get_nr_choices(self):
        choice_cnt = 0
        for x in range(self.model.nr_states):
            choice_cnt += self.model.get_nr_available_actions(x)
        return choice_cnt

    # Check model for invalid or missing components
    def check_model(self):
        if not self.has_transition_matrix():
            raise Exception("The POMDP model does not have a transition matrix")

        if not self.has_state_labels():
            print("No state labels where found in the POMDP model")

        if not self.has_reward_model():
            raise Exception("The PODP model is missing reward models")

        if not self.model.has_choice_labeling():
            raise Exception("The POMDP model does not have choice labels")

        if not self.model.has_state_valuations():
            print("The POMDP model has not state valuations")
    
    def has_transition_matrix(self):
        return self.model.nr_states > 0 \
            and self.model.nr_transitions > 0 \
            and len(self.model.transition_matrix) > 0

    def has_state_labels(self):
        return len(self.model.labeling.get_labels()) > 0

    def has_reward_model(self):
        return True if len(self.model.reward_models) > 0 else False

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

    

