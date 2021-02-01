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
        ### initialize necessary info
        self.d = self.representation.param.shape[0]
        self.A_t_inv = np.eye(self.d)/self.reg_val
        self.b_t = np.zeros(self.d)
        self.theta_t = np.zeros(self.d)
        ###################################
        self.t = 1

    def sample_action(self, context):
        ### implement action selection strategy
        phi = self.representation.features[context]
        alpha_t = self.noise_std*sqrt(self.d*log((1+self.t*self.param_bound**2/self.reg_val)/self.delta)) + sqrt(self.reg_val)*self.features_bound
        mu_t = phi @ self.theta_t
        B_t = mu_t + alpha_t*np.sqrt(np.diag(phi @ self.A_t_inv @ phi.T))
        maxa = np.argmax(B_t)
        ###################################
        self.t += 1
        return maxa

    def update(self, context, action, reward):
        v = self.representation.get_features(context, action)
        ### update internal info (return nothing)
        # Sherman–Morrison formula
        self.A_t_inv = self.A_t_inv - (self.A_t_inv @ v[:, None] @ v[None, :] @ self.A_t_inv)/(1 + v[None, :] @ self.A_t_inv @ v[:, None])
        self.b_t += reward * v
        self.theta_t = self.A_t_inv @ self.b_t
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



