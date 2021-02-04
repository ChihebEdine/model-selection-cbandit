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
        self.d = self.representation.dim()
        self.A_t_inv = np.eye(self.d)/self.reg_val
        self.b_t = np.zeros(self.d)
        self.theta_t = np.zeros(self.d)
        self.regret_bound = 0
        ###################################
        self.t = 1

    def sample_action(self, context):
        ### implement action selection strategy
        phi = self.representation.features[context]
        beta_t = self.noise_std * sqrt( - log(np.linalg.det(self.A_t_inv)*(self.reg_val**self.d)*(self.delta**2))) + sqrt(self.reg_val)*self.param_bound
        #beta_t = self.noise_std*sqrt(self.d*log((1+self.t*self.features_bound**2/self.reg_val)/self.delta)) + sqrt(self.reg_val)*self.param_bound
        mu_t = phi @ self.theta_t
        norms = np.sqrt(np.diag(phi @ self.A_t_inv @ phi.T))
        B_t = mu_t + beta_t*norms
        maxa = np.argmax(B_t)
        self.regret_bound += 2*beta_t*norms[maxa]
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
        ### initialize necessary info
        self.learners = [LinUCB(r, self.reg_val, self.noise_std, self.delta) for r in self.representations]
        self.U = np.zeros(len(self.representations))
        self.n = np.zeros(len(self.representations))
        self.active_reps = set(range(len(self.representations)))
        ###################################
        self.t = 1
    
    def optimistic_action(self, rep_idx, context):
        ### implement action selection strategy given the selected representation
        maxa = self.learners[rep_idx].sample_action(context)
        ###################################
        return maxa

    def sample_action(self, context):
        ### implement representation selection strategy
        ### and action selection strategy
        rep_idx = min(self.active_reps, key=lambda i: self.learners[i].regret_bound)
        action = self.optimistic_action(rep_idx, context)
        self.last_selected_rep = rep_idx
        ###################################
        self.t += 1
        return action

    def update(self, context, action, reward):
        idx = self.last_selected_rep
        ### implement update of internal info and active set
        self.learners[idx].update(context, action, reward)
        self.U[idx] += reward
        self.n[idx] += 1

        if all(self.n>=2):
            c = 1
            M = len(self.learners)

            regret_bounds = np.array([learner.regret_bound for learner in self.learners])
            upper_bounds = (self.U + regret_bounds)/self.n + c * np.sqrt(np.log(M*np.log(self.n)/self.delta)/self.n)
            lower_bounds = self.U/self.n - c * np.sqrt(np.log(M*np.log(self.n)/self.delta)/self.n)
            max_lower_bound = np.max(lower_bounds[list(self.active_reps)])
            ids_to_eliminate = set(np.where(upper_bounds < max_lower_bound)[0])
            ids_to_eliminate = self.active_reps & ids_to_eliminate

            self.active_reps -= ids_to_eliminate
            if ids_to_eliminate:
                print(f"ids eliminated : {ids_to_eliminate}")
        ###################################
