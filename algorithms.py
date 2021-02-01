import numpy as np 

import math
from math import log, sqrt

class LinUCB:

    def __init__(self, 
        representation,
        reg_val, noise_std, delta=0.01
    ):
        self.representation = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = representation.param_bound
        self.features_bound = representation.features_bound
        self.delta = delta
        self.reset()

    def reset(self):
        ### TODO: initialize necessary info

        ###################################
        self.t = 1

    def sample_action(self, context):
        ### TODO: implement action selection strategy

        ###################################
        self.t += 1
        return maxa

    def update(self, context, action, reward):
        v = self.representation.get_features(context, action)
        ### TODO: update internal info (return nothing)

        ###################################


class RegretBalancingElim:
    def __init__(self, 
        representations,
        reg_val, noise_std,delta=0.01
    ):
        self.representations = representations
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = [r.param_bound for r in representations]
        self.features_bound = [r.features_bound for r in representations]
        self.delta = delta
        self.last_selected_rep = None
        self.active_reps = None # list of active (non-eliminated) representations
        self.t = None
        self.reset()
    

    def reset(self):
        ### TODO: initialize necessary info

        ###################################
        self.t = 1
    
    def optimistic_action(self, rep_idx, context):
        ### TODO: implement action selection strategy given the selected representation

        ###################################
        return maxa

    def sample_action(self, context):
        ### TODO: implement representation selection strategy
        #         and action selection strategy

        ###################################
        self.t += 1
        return action

    def update(self, context, action, reward):
        idx = self.last_selected_rep
        v = self.representations[idx].get_features(context, action)
        ### TODO: implement update of internal info and active set

        ###################################



