import tensorflow as tf
import numpy as np
import cplex
import cdd
from scipy.spatial import ConvexHull
from collections import defaultdict

from utils import *
from ilp import *

from ilp_utils_ex import enum_solutions
from layers import ReluLayer, LinearLayer


def load_tensorflow_graph():
    pb = './model/pretrain_linear_reward.pb'

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(pb, 'rb').read())

    tf.graph_util.import_graph_def(graph_def)
    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.Session()

    return graph, sess

class PropertyExplainer:
    def __init__(self):
        self.graph, self.sess = load_tensorflow_graph()
        self.layers = list()

        self.ilp_vars_info_all_layers = {}
        self.ilpsolver = ILPSolver()

    def encode_input_layers(self):
        ##############################
        # Prev bitrate branch
        ##############################
        w_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((128,))
        b_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        lbs = [0.0]
        ubs = [1.0]
        possible_values = [v / 4300 for v in [300, 750, 1200, 1850, 2850, 4300]]

        branch1 = ReluLayer('prev_bitrate', w, b, 'FC', lbs, ubs, self.ilpsolver, possible_values, ilp_vars_info=self.ilp_vars_info_all_layers)

        self.layers.append(branch1)

        ##############################
        # Buffer branch
        ##############################
        w_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_1/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((128,))
        b_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_1/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        lbs = [0.4]
        ubs = [3.267816]

        branch2 = ReluLayer('buffer', w, b, 'FC', lbs, ubs, self.ilpsolver, ilp_vars_info=self.ilp_vars_info_all_layers)

        self.layers.append(branch2)

        ##############################
        # Remaining chunks branch
        ##############################
        w_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_2/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((128,))
        b_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_2/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        lbs = [0.0]
        ubs = [1.0]
        possible_values = [v / 48 for v in range(48)]

        branch3 = ReluLayer('remaining_chunks', w, b, 'FC', lbs, ubs, self.ilpsolver, possible_values, ilp_vars_info=self.ilp_vars_info_all_layers)

        self.layers.append(branch3)

        ##############################
        # Throughput branch
        ##############################
        w_tensor = self.graph.get_tensor_by_name('import/actor/Conv1D/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((4, 1, 8, 128))
        b_tensor = self.graph.get_tensor_by_name('import/actor/Conv1D/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        slice = 1
        w = w[slice, 0, :, :]

        lbs = [0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0]
        ubs = [0.541045, 0.541045, 0.541045, 0.541045,
               0.541045, 0.542024, 0.542086, 0.542258]

        branch4 = ReluLayer('throughput', w, b, 'Conv', lbs, ubs, self.ilpsolver, ilp_vars_info=self.ilp_vars_info_all_layers)

        self.layers.append(branch4)

        ##############################
        # Latency branch
        ##############################
        w_tensor = self.graph.get_tensor_by_name('import/actor/Conv1D_1/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((4, 1, 8, 128))
        b_tensor = self.graph.get_tensor_by_name('import/actor/Conv1D_1/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        slice = 1
        w = w[slice, 0, :, :]

        lbs = [0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.083082]
        ubs = [1.210133, 1.226670, 1.226670, 1.253752,
               1.257164, 1.259484, 1.259484, 1.259484]

        branch5 = ReluLayer('latency', w, b, 'Conv', lbs, ubs, self.ilpsolver, ilp_vars_info=self.ilp_vars_info_all_layers)

        self.layers.append(branch5)

        ##############################
        # Next size branch
        ##############################
        w_tensor = self.graph.get_tensor_by_name('import/actor/Conv1D_2/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((4, 1, 6, 128))
        b_tensor = self.graph.get_tensor_by_name('import/actor/Conv1D_2/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        slice = 1
        w = w[slice, 0, :, :]

        lbs = [0.111155, 0.277716, 0.487160,
               0.728962, 1.139290, 1.919781]
        ubs = [0.181901, 0.450283, 0.709534,
               1.076598, 1.728878, 2.395588]

        branch6 = ReluLayer('next_size', w, b, 'Conv', lbs, ubs, self.ilpsolver, ilp_vars_info=self.ilp_vars_info_all_layers)

        self.layers.append(branch6)


        self.inputs = list()
        # second layer inputs concat
        self.layers = [self.layers[0], self.layers[1], self.layers[3],
                       self.layers[4], self.layers[5], self.layers[2]]
        for branch in self.layers:
            self.inputs.extend(branch.ilp_vars_info[ILP_OUTPUT_VARS + str(branch.name)])


    def encode_hidden_layers(self):
        ###################################
        # Second Layer
        ###################################
        ## fully connected
        w_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_3/W/read:0')
        w = w_tensor.eval(session=self.sess).reshape((768, 128))
        b_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_3/b/read:0')
        b = b_tensor.eval(session=self.sess).reshape((128,))

        lbs = [0]*len(b)
        ubs = [cplex.infinity]*len(b)
        layer_2_step_1 = ReluLayer('layer_2_step_1', w, b, 'FC', lbs, ubs, self.ilpsolver, ilp_vars_info=self.ilp_vars_info_all_layers, inputs=self.inputs)

        ## output
        wo_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_4/W/read:0')
        wo = wo_tensor.eval(session=self.sess).reshape((128, 6))
        bo_tensor = self.graph.get_tensor_by_name('import/actor/FullyConnected_4/b/read:0')
        bo = bo_tensor.eval(session=self.sess).reshape((6,))


        lbs = [0]*len(bo)
        ubs = [cplex.infinity]*len(bo)

        layer_2_step_2 = LinearLayer('layer_2_step_2', wo, bo, 'FC', lbs, ubs, self.ilpsolver, ilp_vars_info=self.ilp_vars_info_all_layers, inputs=layer_2_step_1.output_names)

        self.hidden_layers = [layer_2_step_1, layer_2_step_2]


    def encode_constant_values(self, default_values):
        for feature in default_values:
            vars = [feature]
            coefs = [1]
            rhs = [default_values[feature]]
            tag_con = "fixed_feature_value_" + str(feature)

            self.ilpsolver.add_linear_constraint(rhs=rhs,
                                                 senses='E',
                                                 vars=vars,
                                                 coefs=coefs,
                                                 tag=tag_con)


    def encode_tighter_bounds(self, variable, lb, ub):
        vars = [variable]
        coefs = [1]

        rhs = [lb]
        tag_con = "constraint_lower_bound_" + str(variable)
        self.ilpsolver.add_linear_constraint(rhs=rhs,
                                             senses='G',
                                             vars=vars,
                                             coefs=coefs,
                                             tag=tag_con)

        rhs = [ub]
        tag_con = "constraint_upper_bound_" + str(variable)
        self.ilpsolver.add_linear_constraint(rhs=rhs,
                                             senses='L',
                                             vars=vars,
                                             coefs=coefs,
                                             tag=tag_con)


    def encode_expected_output(self, expected_br, inverted=False):
        # Add constraint: selected bitrate = expected
        # (Return violation if this bit rate is found)
        # If reverted == True, return violation is any other bit rate is found
        self.expected_br = expected_br

        prop_bin_input_names, _ = self.ilpsolver.add_variables(
                label="q",
                ids=['prop_selected_output'],obj=0,
                lb=0,
                ub=1,
                type=self.ilpsolver.solver.variables.type.binary,
                sz=6)
        self.ilpsolver.store_ilp_var_info(
                self.ilp_vars_info_all_layers,
                ILP_PROP_BIN_INPUT_VARS + str('prop_selected_out'),
                prop_bin_input_names)

        self.output_indicator_vars = prop_bin_input_names

        for i in range(6):
            if i == expected_br:
                continue
            ind_vars = prop_bin_input_names[i]
            vars = [self.hidden_layers[1].output_names[expected_br],
                    self.hidden_layers[1].output_names[i]]
            rhs = 0
            coefs = [1, -1]
            tag_con = "prop_out_ind_" + str(i)

            self.ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                                    vars=vars,
                                                    coefs=coefs,
                                                    rhs=rhs,
                                                    senses='G',
                                                    complemented=0,
                                                    tag=tag_con)

            coefs = [-1, 1]
            tag_con = "prop_out_ind_reverse_" + str(i)

            self.ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                                    vars=vars,
                                                    coefs=coefs,
                                                    rhs=rhs,
                                                    senses='G',
                                                    complemented=1,
                                                    tag=tag_con)
        # Violation iff there is no b_i > b_2
        coefs = [1,1,1,1,1,1]
        coefs[expected_br] = 0
        vars = prop_bin_input_names
        if not inverted:
            rhs = 5
            senses = 'E'
        else:
            rhs = 4.9
            senses = 'L'
        tag_con = "prop_set_var_{}_sum_greater_than_1".format(self.hidden_layers[1].output_names[expected_br])

        first_constraint = [vars, coefs]
        rhs = [rhs]
        constraint_names = [tag_con]
        constraints = [first_constraint]
        self.ilpsolver.add_linear_constraint(rhs=rhs,
                                             senses=senses,
                                             vars=vars,
                                             coefs=coefs,
                                             tag=tag_con)


    def find_solutions(self, stop_after_one=False):
        self.ilpsolver.solver.write("test_pairwise", "lp")

        bin_names_all_layers = list()
        for branch in self.layers: # layer 1
            bin_names_all_layers.extend(branch.ilp_vars_info[ILP_Z_VARS + str(branch.name)])
        bin_names_all_layers.extend(self.ilp_vars_info_all_layers[ILP_Z_VARS + str(self.hidden_layers[0].name)]) # layer 2
        self.relu_states = bin_names_all_layers

        all_vars = set()
        for layer in self.layers + self.hidden_layers:
            for var in layer.input_names + layer.output_names:
                all_vars.add(var)
        self.vars_return = list(all_vars)
        self.vars_return += self.relu_states

        #self.vars_return += self.output_indicator_vars

        if stop_after_one:
            stop_cond = lambda x,y: True
            self.ilpsolver.solver.parameters.mip.pool.intensity.set(1)
            self.ilpsolver.solver.parameters.mip.limits.populate.set(1)
        else:
            stop_cond = None

        self.ilpsolver.solver.write('test_module', 'lp')

        self.solutions, self.vars_values = enum_solutions(
                self.ilpsolver,
                block_vars_names=bin_names_all_layers,
                vars_print=[],
                vars_return=self.vars_return,
                stop_cond=stop_cond, verbose=False)


    def find_assignments(self, sol_idx):
        sol = self.solutions[sol_idx]
        vars = self.vars_values[sol_idx]

        vars_values = {var: val for var, val in zip(self.vars_return, vars)}
        return vars_values


    def find_linear_equation(self, sol_idx, free_vars):
        vars_values = self.find_assignments(sol_idx)

        params = dict()

        # record constant inputs
        for layer in self.layers:
            for var in layer.input_names:
                if var not in free_vars:
                    params[var] = defaultdict(int, {'c': vars_values[var], 'relu': 1})
                else:
                    params[var] = defaultdict(int, {'relu': 1})

        # record non-constant relations
        for layer in self.layers + self.hidden_layers[0:1]:
            inputs = layer.input_names
            outputs = layer.output_names
            relus = layer.relu_state_names
            weights = layer.w
            biases = layer.b

            for var, relu, w, b in zip(outputs, relus, weights.T, biases):
                w = np.reshape(w, (-1,))
                # look at each outputs
                params[var] = defaultdict(int)
                relu_value = vars_values[relu]

                # look at weight of each input
                for input_w, input_name in zip(w, inputs):
                    params[var][input_name] = input_w
                params[var]['c'] = b
                params[var]['relu'] = int(relu_value)

        # record output variables
        inputs = self.hidden_layers[1].input_names
        outputs = self.hidden_layers[1].output_names
        weights = self.hidden_layers[1].w
        biases = self.hidden_layers[1].b

        for var, w, b in zip(outputs, weights.T, biases):
            w = np.reshape(w, (-1,))
            # look at each outputs
            params[var] = defaultdict(int)
            relu_value = vars_values[relu]

            # look at weight of each input
            for input_w, input_name in zip(w, inputs):
                params[var][input_name] = input_w
            params[var]['c'] = b
            params[var]['relu'] = -1 # invalid value, indicates that no relu

        return params


    def find_simplified_params(self, params, free_vars):
        '''Version "find the polytope"'''
        # propagate all values to rely only on constants and free variables
        simplified = set()
        def _simplify_var_params(var):
            # do not try to simplify again
            if var in simplified:
                return
            # go through all the subvars
            for var2 in list(params[var].keys()):
                # skip keys that are not bound parameters
                if var2 in free_vars or var2 == 'c' or var2 == 'relu':
                    continue
                # recursively simplify first
                _simplify_var_params(var2)
                # if the relu is off, the contribution is 0, delete immediately
                if params[var2]['relu'] != 0:
                    for var3 in params[var2]:
                        if var3 == 'relu':
                            continue
                        params[var][var3] += params[var][var2]*params[var2][var3]
                del params[var][var2]
            simplified.add(var)

        for var in params:
            _simplify_var_params(var)


    def find_constraints(self, params, free_vars):
        constr = list()
        v1, v2 = free_vars

        # relu constraints
        for var in params:
            if params[var]['relu'] == -1:
                # no relu in this layer, does not yield relu constraints
                continue
            elif params[var]['relu'] == 0:
                # coef1*v1 + coef2*v2 + const <= 0
                constr.append((-params[var][v1], -params[var][v2], -params[var]['c']))
            else:
                # coef1*v1 + coef2*v2 + const > 0
                constr.append((params[var][v1], params[var][v2], params[var]['c']))

        # output constraints
        if hasattr(self, 'expected_br'):
            out_e = self.hidden_layers[1].output_names[self.expected_br] # output expected
            for out_i in self.hidden_layers[1].output_names: # output iterated
                if out_i == out_e:
                    continue
                else:
                    # out_e > out_i
                    # coef1*v1_e + coef2*v2_e + const_e > coef1*v1_i + coef2*v2_i + const_i
                    # (coef1_e - coef1_i)*x1 + (coef2_e - coef2_i)*x2 + (const_e - const_i) > 0
                    constr.append((params[out_e][v1] - params[out_i][v1],
                                   params[out_e][v2] - params[out_i][v2],
                                   params[out_e]['c'] - params[out_i]['c']))


        return constr


    def find_polytope(self, constr):
        mat = np.array(constr)

        # constant must be first
        mat = np.hstack((mat[:, -1:], mat[:, :-1]))
        # `mat` represents one polyhedron to exclude
        # e.g. mat[0, :, :] has shape (384, 3) and corresponds to 384 inequations
        mat = cdd.Matrix(mat, number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        mat.canonicalize()

        poly = cdd.Polyhedron(mat)
        ext = poly.get_generators()

        extreme_points = []
        for e in ext:
            e = np.asarray(e)
            e = np.delete(e, 0)
            extreme_points.append(e)

        extreme_points = np.asarray(extreme_points)

        hull = ConvexHull(extreme_points)
        points = extreme_points[hull.vertices]
        points = np.vstack((points, points[0]))

        return points
