import os
import stormpy
import stormpy.pomdp
from pmc import PMC

def solve_pomdp(model, properties, memory_size=2, 
                undefined_constants=None, action_reward=None, pmc_parameters=None):
    pomdp = POMDP()
    pomdp.load_model(model)
    pomdp.parse_properties(properties)
    if not undefined_constants == None:
        pomdp.set_undefined_constants(undefined_constants)

    pomdp.build_model()
    if not action_reward == None:
        reward_models = {}
        reward_models[''] = stormpy.SparseRewardModel(optional_state_action_reward_vector=action_reward)
        components = stormpy.SparseModelComponents(transition_matrix=pomdp.model.transition_matrix\
        , state_labeling=pomdp.model.labeling, reward_models=reward_models)
    
        components.choice_labeling = pomdp.model.choice_labeling
        components.observability_classes = pomdp.model.observations
        pomdp.model = stormpy.storage.SparsePomdp(components)

    pmc = PMC(pomdp.build_pmc(nr_memory_state=memory_size))
    pmc.export_parametric_to_drn("test.out")
    
    pmc.instantiate_parameters([0.4 for _ in range(pmc.nr_parameters)])
    pmc.model_checking(pomdp.properties[0])
    pmc.print_results()


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
            raise Exception("Model file type not supported to parse properties")      

    # constants is a String containing a variable name and its value(e.g. "K=4, T=2")
    def set_undefined_constants(self, constants):
        desc = stormpy.SymbolicModelDescription(self.program)
        const = desc.parse_constant_definitions(constants)
        if self.file_type == ".prism":
            self.program = desc.instantiate_constants(const).as_prism_program()
        elif self.file_type == ".jani":
            self.program = desc.instantiate_constants(const).as_jani_model()
        else:
            raise Exception("Model file type not supported to define constants")    

    def build_model(self):
        if self.file_type == ".prism" or self.file_type == ".jani":
            self.model = stormpy.build_model(self.program, self.properties)
        elif self.file_type == ".drn":
            parser_options = stormpy.DirectEncodingParserOptions()
            parser_options.build_choice_labels = True
            self.model = stormpy.build_model_from_drn(self.file_path, parser_options)
        else:
            raise Exception("Model file type not supported to build models") 

    def build_pmc(self,\
        pomdp_memory_pattern=stormpy.pomdp.PomdpMemoryPattern.selective_counter,\
        nr_memory_state=2,\
        fsc_application_mode=stormpy.pomdp.PomdpFscApplicationMode.simple_linear):
        
        # Checking model to determine if it incomplete.  
        self.inspect_model()

        self.model = stormpy.pomdp.make_canonic(self.model)
        self.model = stormpy.pomdp.make_simple(self.model)
        memoryBuilder = stormpy.pomdp.PomdpMemoryBuilder()
        memory = memoryBuilder.build(pomdp_memory_pattern, nr_memory_state)
        self.model = stormpy.pomdp.unfold_memory(self.model, memory)

        return stormpy.pomdp.apply_unknown_fsc(self.model, fsc_application_mode)

    # Inspect model for invalid or missing components
    def inspect_model(self):
        if not self.has_transition_matrix():
            raise Exception("The POMDP model does not have a transition matrix")

        if not self.has_state_labels():
            print("No state labels where found in the POMDP model")

        if not self.has_reward_model():
            raise Exception("The PODP model is missing reward models")

        if not self.model.has_choice_labeling():
            raise Exception("The POMDP model does not have choice labels")

        if not self.model.has_state_valuations():
            print("The POMDP model has no state valuations")
    
    def has_transition_matrix(self):
        return self.model.nr_states > 0 \
            and self.model.nr_transitions > 0 \
            and len(self.model.transition_matrix) > 0

    def has_state_labels(self):
        return len(self.model.labeling.get_labels()) > 0

    def has_reward_model(self):
        return True if len(self.model.reward_models) > 0 else False
    
    def has_undefined_constants(self):
        return self.program.has_undefined_constants

    # Get functions
    def get_model_type(self):
        return self.model.model_type

    def get_states(self):
        return self.model.states
    
    def get_nr_states(self):
        return len(self.model.states)

    def get_state_labels(self):
        return self.model.labeling.get_labels()

    def get_choice_labels(self):
        return self.model.choice_labeling.get_labels()
    
    def get_reward_models(self):
        return self.model.reward_models
    
    def get_observations(self):
        return self.model.observations

    # Print functions
    def print_model_type(self):
        print("Model type:", self.get_model_type())

    def print_nr_states(self):
        print("Number of states:", self.get_nr_states())
    
    def print_actions(self):
        for state in self.model.states:
            print("State", state, " has actions", [str(action) for action in state.actions])

    def print_transitions(self):
        for state in self.model.states:
            for action in state.actions:
                for transition in action.transitions:
                    print("State", state, " using action", action," goes to state",\
                         transition.column," with propability", transition.value())
    
    def print_observations(self):
        for i in range(self.get_nr_states()):
            print("State", i, " has observation", self.model.get_observation(i))
    
    def print_rewards(self):
        for key in self.model.reward_models:
            reward_model = self.model.reward_models[key]
            if reward_model.has_state_rewards:
                self.print_state_reward(reward_model.state_rewards)
            if reward_model.has_state_action_rewards:
                self.print_state_action_reward(reward_model.state_action_rewards)
            if reward_model.has_transition_rewards:
                print("has transition rewards")

    def print_state_reward(self, reward):
        print("***** State Rewards *****")
        i = 0
        for state in self.get_states():
            for action in state.actions:
                print("State", state, " has a reward of", reward[i])
                i += 1

    def print_state_action_reward(self, reward):
        print("***** State Action Rewards *****")
        i = 0
        for state in self.get_states():
            for action in state.actions:
                print("State", state, " using action", action, " has a reward of", reward[i])
                i += 1
    
    def print_constants(self):
        for c in self.program.constants:
            if c.defined:
                print("Constant:", c.name, " type:", c.type, " Definition:", c.definition)
            else:
                print("Constant:", c.name, " type:", c.type, " Definition: Undefined")



    

