import os
import json
import logging
import argparse
import sys  
import numpy as np
from utils import *
import random
from scipy.spatial import ConvexHull

from math import floor, fabs

print("cplex")
import cplex
from cplex.exceptions import CplexError    
import cplex.callbacks as CPX_CB

class ILPSolver:
    def __init__(self, file_name = None):
        if file_name is None:
            self.solver = cplex.Cplex()
        else:
            self.solver = cplex.Cplex(file_name)
    
        self.inputs = []
        self.states = []
        self.outputs = []
        self.choice = []
                
    def store_ilp_var_info(self, ilp_dict, label, vars_names, is_special_vars = None):
        ilp_dict[label] = vars_names
        if (is_special_vars == INPUT_VARS_TAG):
            self.inputs.append(np.asarray(vars_names).copy())
        if (is_special_vars == STATE_VARS_TAG):
            self.states.append(np.asarray(vars_names).copy())
        if (is_special_vars == OUTPUT_VARS_TAG):
            self.outputs.append(np.asarray(vars_names).copy())        
        if (is_special_vars == CHOICE_VAR_TAG):
            self.choice = np.asarray(vars_names).copy()
        
        
            
               
    def form_var_name(self, tag, ids):
        for id in ids:
            tag = tag + "_" + str(id)
        return tag+"$"
    
    def add_variables(self, label, ids,  obj, lb, ub, type, sz):        
        names = [""] * sz
        for i in range(sz):
           names[i] = self.form_var_name(label,   np.append(ids, [i]))
            
        objective = [obj] * sz
        lower_bounds = [lb] * sz         
        upper_bounds = [ub] * sz
        types = [type] * sz
        
        #print(types) 
        inds = self.solver.variables.add( obj =objective,
                              lb =lower_bounds,
                              ub =upper_bounds,
                              types = types,
                              names = names) 
        return names, inds
    
    def add_variables_vec (self, label, ids,  obj, lb, ub, type, sz):        
        names = [""] * sz
        for i in range(sz):
           names[i] = self.form_var_name(label,   np.append(ids, [i]))

        if (np.isscalar(obj)):            
            objective = [obj] * sz
        else:
            objective =  [o for o in obj]
        if (np.isscalar(lb)):            
            lower_bounds = [lb] * sz         
        else:        
            lower_bounds = [float(i) for i in lb]       

        if (np.isscalar(ub)):            
            upper_bounds = [ub] * sz         
        else:                        
            upper_bounds = [float(i) for i in ub] 

        if (np.isscalar(type)):            
            types = [type] * sz         
        else:                                    
            types = [t for t in type]

        inds = self.solver.variables.add( obj =objective,
                              lb =lower_bounds,
                              ub =upper_bounds,
                              names = names) 
        return names, inds
    def add_variables_vec_obj(self, label, ids,  obj, lb, ub, type, sz):        
        names = [""] * sz
        for i in range(sz):
           names[i] = self.form_var_name(label,   np.append(ids, [i]))
            
        objective = [float(i) for i in obj]       
        lower_bounds = [lb] * sz         
        upper_bounds = [ub] * sz
        types = [type] * sz
        
         
        self.solver.variables.add( obj =objective,
                              lb =lower_bounds,
                              ub =upper_bounds,
                              names = names) 
        return names
    
    def add_linear_constraint(self, vars, coefs, rhs, senses, tag):        
        self.solver.linear_constraints.add( rhs = rhs,
                                        senses = [senses],
                                        lin_expr = [[vars, coefs]],
                                        names = [tag])
        
    def add_indicator_constraint(self, indicator_vars, vars, coefs, rhs, senses, complemented, tag):        
#         print('indicator_vars',  indicator_vars)
#         print('vars',  vars)
#         print("coefs", coefs)
#         print("rhs", rhs)        
#         print("tag", tag)        
        #exit()
        self.solver.indicator_constraints.add(indvar = indicator_vars,
                       complemented = complemented,
                       rhs = rhs,
                       sense = senses,
                       lin_expr = [vars, coefs],
                       name = tag)   
    def equvalence_indicator_constraint(self, indicator_vars, vars, coefs, rhs, senses, complemented, tag):
        self.add_indicator_constraint(indicator_vars, vars, coefs, rhs, senses[0], complemented[0], tag)
        self.add_indicator_constraint(indicator_vars, vars, coefs, rhs-0.001, senses[1], complemented[1], tag)
        
        
    def delete_linear_constraints(self, tag):
        self.solver.linear_constraints.delete(tag)