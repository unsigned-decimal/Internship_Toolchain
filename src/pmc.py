import os
import stormpy
import stormpy.pomdp
import stormpy.pars
from pycarl.cln.cln import Rational
import time
class PMC:

    def __init__(self, model):
        self.model = model
        self.nr_parameters = len(self.model.collect_probability_parameters())
        self.result = None  
    
    def set_parameters(self, parameter_values):
        parameters = list(self.model.collect_probability_parameters())
        if len(parameter_values) == len(parameters):
            instantiator = stormpy.pars.PDtmcInstantiator(self.model)
            par_dict = dict()
            for x in range(len(parameters)):
                parameter_x = parameters[x]
                par_dict[parameter_x] = Rational(parameter_values[x])
            self.model = instantiator.instantiate(par_dict)
        else:
            raise Exception("Number of parameter values does not match number of parameters, expected:"\
                , len(parameters), ", found:", len(parameter_values)) 


    def model_checking(self, property):
        self.result = stormpy.model_checking(self.model, property)
    
    def get_results(self):
        #Qualitative result types do not have a get_values() method 
        #but instead a get_truth_values() method
        qualitative_result_types = [stormpy.ExplicitQualitativeCheckResult, \
            stormpy.SymbolicQualitativeCheckResult]
        
        #These type do not have a get_values() or a get_truth_values() method
        not_supported_result_types = [stormpy.SymbolicParametricQuantitativeCheckResult, \
            stormpy.SymbolicExactQuantitativeCheckResult, stormpy.SymbolicQuantitativeCheckResult]

        check_result_type = lambda type: isinstance(self.result, type)

        if any(map(check_result_type, not_supported_result_types)):
            raise Exception("Result type not supported")
        elif any(map(check_result_type, qualitative_result_types)):
            return self.result.get_truth_values()
        else:
            return self.result.get_values() 

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
        for x in self.get_results():
            print("Result", cnt," =", x)
            cnt += 1

    #TODO
    #get_parameters
    #print_parameters
    #get_nr_parameters
    #additional tests
    

