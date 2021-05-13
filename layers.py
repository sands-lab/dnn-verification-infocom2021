import numpy as np

from ilp import *
from ilp_utils import enum_solutions

epsilon = 0.0001


class AbstractLayer:
    def __init__(self, name, w, b, op, lbs, ubs, ilpsolver, possible_values=list(), ilp_vars_info=None, inputs=None):
        """
        Class for one network layer.
        :lbs: and :ubs: are matrices with the same size as the input
        :op: is either 'FC' or 'Conv'
        :w: and :b: are the weights / filters and biases
        """
        self.name = name
        self.op = op

        self.w = np.array(w)
        self.b = np.array(b)


        self.lbs = np.array(lbs)
        self.ubs = np.array(ubs)

        assert self.lbs.shape == self.ubs.shape
        self.possible_values = possible_values

        self.nb_inputs = len(self.lbs)
        self.nb_outputs = len(b)
        self.ilp_vars_info, self.input_names, self.bin_input_names, self.output_names = self._create_vars(ilpsolver, ilp_vars_info, inputs)
        self._encode_layer(ilpsolver)


    def solve(self, ilpsolver):
        self.bin_vars = self.ilp_vars_info[ILP_Z_VARS + str(self.name)]
        self.solutions = enum_solutions(ilpsolver, self.bin_vars, self.bin_vars)
        self.solutions = [list(map(round, sol)) for sol in self.solutions]
        self.intervals, self.output_boxes = self._compute_boxes()

        self.intervals, self.solutions, self.output_boxes = \
                zip(*sorted(zip(self.intervals, self.solutions, self.output_boxes),
                            key=lambda x: np.array(x[0]).flatten()[0]))


    def _test_solve(self, ilpsolver):
        try:
            ilpsolver.solver.write("test_check", "lp")
            print("solving to begin")
            ilpsolver.solver.solve()
            print("solved")
            if (ilpsolver.solver.solution.get_status() == ilpsolver.solver.solution.status.MIP_infeasible):
                print("Layer {}: no solutions".format(layer))
            if (ilpsolver.solver.solution.is_primal_feasible()):
                solutions = ilpsolver.solver.solution
            else:
                print("Layer {}: terminated".format(layer))
        except CplexError as e:
            print("Layer {}: Exception raised during solving: {}".format(layer, e))


    def _create_vars(self, ilpsolver, ilp_vars_info=None, inputs=None):
        if (ilp_vars_info is None):
            ilp_vars_info = {}
        #########################################################
        # create input variables
        #########################################################
        if (inputs is None):
            input_names, _ = ilpsolver.add_variables_vec(label="x",
                                                         ids=[self.name],
                                                         obj=0,
                                                         lb=self.lbs,
                                                         ub=self.ubs,
                                                         type=ilpsolver.solver.variables.type.continuous,
                                                         sz=self.nb_inputs)
            is_special_vars = INPUT_VARS_TAG
            ilpsolver.store_ilp_var_info(ilp_vars_info,
                                         ILP_INPUT_VARS + str(self.name),
                                         input_names,
                                         is_special_vars=is_special_vars)
        else:
            input_names = inputs
            is_special_vars = INPUT_VARS_TAG
            ilpsolver.store_ilp_var_info(ilp_vars_info,
                                         ILP_INPUT_VARS + str(self.name),
                                         input_names,
                                         is_special_vars=is_special_vars)

        if len(self.possible_values) > 0:
            bin_input_names, _ = ilpsolver.add_variables(label="v",
                                                      ids=[self.name],obj=0,
                                                      lb=0,
                                                      ub=1,
                                                      type=ilpsolver.solver.variables.type.binary,
                                                      sz=len(self.possible_values))
            ilpsolver.store_ilp_var_info(ilp_vars_info,
                                         ILP_BIN_INPUT_VARS + str(self.name),
                                         bin_input_names)
        else:
            bin_input_names = None

        #########################################################
        # create output variables
        #########################################################

        output_names, _ = ilpsolver.add_variables(label="s",
                                                  ids=[self.name],
                                                  obj=0,
                                                  lb=0,
                                                  ub=cplex.infinity,
                                                  type=ilpsolver.solver.variables.type.continuous,
                                                  sz=self.nb_outputs)

        is_special_vars = OUTPUT_VARS_TAG
        ilpsolver.store_ilp_var_info(ilp_vars_info,
                                     ILP_OUTPUT_VARS + str(self.name),
                                     output_names,
                                     is_special_vars=is_special_vars)

        return ilp_vars_info, input_names, bin_input_names, output_names


    def _encode_layer(self):
        raise NotImplementedError('This abstract method must be implemented.')


    def _compute_boxes(self):
        if self.op == 'FC':
            return self._compute_boxes_fc()
        elif self.op == 'Conv':
            return self._compute_boxes_conv()


    def _compute_boxes_fc(self):
        cuts = list()
        boxes = list()
        for j, sol in enumerate(self.solutions):
            intervals = []
            output_box = []
            for i, v in enumerate(sol):
                w = self.w[i]
                b = self.b[i]
                if (v  == 0):
                    if(w >= 0):
                        ub_cut = min(-b/w, self.ubs[0])
                        intervals.append([self.lbs[0], ub_cut])
                    else:
                        lb_cut = max(-b/w, self.lbs[0])
                        intervals.append([lb_cut, self.ubs[0]])
                else:
                    if(w >= 0):
                        lb_cut = max(-b/w, self.lbs[0])
                        intervals.append([lb_cut, self.ubs[0]])
                    else:
                        ub_cut = min(-b/w, self.ubs[0])
                        intervals.append([self.lbs[0], ub_cut])

            lbs, ubs = zip(*intervals)
            extreme_lb, extreme_ub = max(lbs), min(ubs)
            for i, v in enumerate(sol):
                w = self.w[i]
                b = self.b[i]

                if (v  == 0):
                    output_box.append([0,0])
                else:
                    if(w >= 0):
                        output_lb = w*extreme_lb + b
                        output_ub = w*extreme_ub + b
                    else:
                        output_lb = w*extreme_ub + b
                        output_ub = w*extreme_lb + b
                    output_box.append([output_lb,output_ub])

            cuts.append((max(lbs), min(ubs)))

            output_box = list(map(np.array, zip(*output_box)))
            boxes.append(output_box)
        # order the cuts
        cuts, boxes = zip(*sorted(zip(cuts, boxes)))

        return cuts, boxes


    def _compute_boxes_conv(self):
        def compute_polytope_vertices(A, b):
            b = b.reshape((b.shape[0], 1))
            mat = cdd.Matrix(np.hstack([b, -A]), number_type='float')
            mat.rep_type = cdd.RepType.INEQUALITY
            P = cdd.Polyhedron(mat)
            g = P.get_generators()
            V = np.array(g)
            vertices = []
            for i in range(V.shape[0]):
                if V[i, 0] != 1:  # 1 = vertex, 0 = ray
                    raise Exception("Polyhedron is not a polytope")
                elif i not in g.lin_set:
                    vertices.append(V[i, 1:])
            return vertices

        all_vertices = []
        non_active_ind = np.where(np.array(self.lbs) == np.array(self.ubs))[0]
        non_active_values = np.array(self.lbs)[non_active_ind]
        active_ind = np.where(np.array(self.lbs) != np.array(self.ubs))[0]

        for j, sol in enumerate(self.solutions):
            non_zeros = 0
            I_A = []
            I_b = []
            D_A = np.identity(len(active_ind))
            # x \leq ub
            # -x \leq -lb
            for i in range(len(active_ind)):
                ai = active_ind[i]
                I_A.append(D_A[i])
                I_b.append(self.ubs[ai])

                I_A.append(-D_A[i])
                I_b.append(-self.lbs[ai])

            # I_A x \leq b
            for i, v in enumerate(sol):
                #print(f"{bin_vars[i]} = {v}", end = " ")
                w = self.w[active_ind, i]
                b = self.b[i]

                # adjust b using fixed inputs

                fixed_part = sum(self.w[non_active_ind, i] * non_active_values)
                b = b + fixed_part
                #print(w,b)

                if (v == 0):
                    # wx + b  <= 0
                    # wx   <= -b
                    I_A.append(w)
                    I_b.append(-b)
                else:
                    # wx + b  >= 0
                    # -wx  -b  <=0
                    # -wx   <= b
                    I_A.append(-w)
                    I_b.append(b)
                    non_zeros += 1

            I_A = np.asarray(I_A)
            I_b = np.asarray(I_b)
            vertices = compute_polytope_vertices(I_A, I_b)
            all_vertices.append(vertices)

        all_mapped_vertices = []
        for j, sol in enumerate(self.solutions):
            vertices = all_vertices[j]
            mapped_vertices = []

            for i, v in enumerate(sol):
                w = self.w[active_ind, i]
                b = self.b[i]

                # adjist b using fixed inputs
                fixed_part = sum(self.w[non_active_ind, i]*non_active_values)
                b = b + fixed_part

                if (v == 0):
                    p = []
                    p = np.zeros(len(vertices))
                else:
                    p = []
                    for ver in vertices:
                        p.append(sum(w*ver) + b)

                mapped_vertices.append(np.asarray(p))
            mapped_vertices = np.asarray(mapped_vertices)
            mapped_vertices = [mapped_vertices[:, i] for i in range(mapped_vertices.shape[1])]

            all_mapped_vertices.append(mapped_vertices)

        return all_vertices, all_mapped_vertices


    def approximate(self, num_polytopes=None, points_per_polytope=None):
        if num_polytopes is not None:
            cut_indices = [int(i * len(self.solutions) / num_polytopes) for i in range(num_polytopes+1)]
        elif points_per_polytope is not None:
            cut_indices = list()
            count = 0
            for i, box in enumerate(self.output_boxes):
                if count + len(box) > points_per_polytope:
                    cut_indices.append(i)
                    count = 0
                count += len(box)
            cut_indices.append(len(self.solutions))
        else:
            raise ValueError('Must provide num_polytopes or points_per_polytope')

        self.approx_solutions = list()
        for i in range(len(cut_indices)-1):
            sols = self.solutions[cut_indices[i]:cut_indices[i+1]]
            boxes = self.output_boxes[cut_indices[i]:cut_indices[i+1]]
            points = list()
            for box in boxes:
                for point in box:
                    point = list(map(lambda x: round(x, 5), point))
                    if point not in points:
                        points.append(point)
            A, B = approximate_points(points)
            self.approx_solutions.append((sols, (A, B)))

        return self.approx_solutions


class LinearLayer(AbstractLayer):
    def _encode_layer(self, ilpsolver):
        input_idx = output_idx = self.name
        input_vars = self.ilp_vars_info[ILP_INPUT_VARS + str(input_idx)]
        output_vars = self.ilp_vars_info[ILP_OUTPUT_VARS + str(output_idx)]

        tag = "neuron"
        for i in range(self.nb_outputs):
            #sum a_ij input_i + b_j = output
            #sum a_ij input_i - output = -b_j
            vars = np.concatenate([input_vars, [output_vars[i]]]) # np.append(x, [o[i], y[i]])
            rhs = [float(-self.b[i])]
            if self.op == 'FC':
                if (len(self.w[i].shape) == 0):
                    coefs = np.concatenate([[self.w[i]], [-1]])
                else:
                    coefs = np.concatenate([self.w[:,i], [-1]])
            tag_con = tag + "_linear" + "_" + str(i) + "_" + str(output_idx)

            #print(coefs.shape)
            #print(vars.shape)
            ilpsolver.add_linear_constraint(rhs=rhs,
                                            senses='E',
                                            vars=vars,
                                            coefs=coefs,
                                            tag=tag_con)

        # Constraints for discrete inputs
        if self.bin_input_names is not None:
            for i in range(len(self.possible_values)):
                #v = 1 → x = possible_values[i]
                ind_vars = self.bin_input_names[i]
                vars = [input_vars[0]]
                rhs = self.possible_values[i]
                coefs = [1]
                tag_con = tag + "_ind_discrete_" + str(i) + "_" + str(output_idx)

                ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                                   vars=vars,
                                                   coefs=coefs,
                                                   rhs=rhs,
                                                   senses='E',
                                                   complemented=0,
                                                   tag=tag_con)

            vars = np.array(self.bin_input_names)
            coefs = [1 for _ in vars]
            rhs = [1]
            tag_con = tag + "_plus_discrete_" + str(output_idx)

            ilpsolver.add_linear_constraint(rhs=rhs,
                                            senses='E',
                                            vars=vars,
                                            coefs=coefs,
                                            tag=tag_con)




class ReluLayer(AbstractLayer):
    def _create_vars(self, ilpsolver, ilp_vars_info=None, inputs=None):
        ilp_vars_info, input_names, bin_input_names, output_names = super()._create_vars(
                ilpsolver, ilp_vars_info=ilp_vars_info, inputs=inputs)
        #form y variables
        y_names, _ = ilpsolver.add_variables(label="y",
                                             ids=[self.name],
                                             obj=0,
                                             lb=0,
                                             ub=cplex.infinity,
                                             type = ilpsolver.solver.variables.type.continuous,
                                             sz=self.nb_outputs)
        ilpsolver.store_ilp_var_info(ilp_vars_info,
                                     ILP_Y_VARS + str(self.name),
                                     y_names)
        #form z variables
        z_names, z_inds = ilpsolver.add_variables(label="z",
                                                  ids=[self.name],obj=0,
                                                  lb=0,
                                                  ub=1,
                                                  type=ilpsolver.solver.variables.type.binary,
                                                  sz=self.nb_outputs)
        ilpsolver.store_ilp_var_info(ilp_vars_info,
                                     ILP_Z_VARS + str(self.name),
                                     z_names)

        self.relu_state_names = z_names

        return ilp_vars_info, input_names, bin_input_names, output_names


    def _encode_layer(self, ilpsolver):
        input_idx = output_idx = self.name
        y = self.ilp_vars_info[ILP_Y_VARS + str(output_idx)]
        z = self.ilp_vars_info[ILP_Z_VARS + str(output_idx)]
        input_vars = self.ilp_vars_info[ILP_INPUT_VARS + str(input_idx)]
        output_vars = self.ilp_vars_info[ILP_OUTPUT_VARS + str(output_idx)]

        tag = "neuron"
        for i in range(self.nb_outputs):
            #sum a_ij input_i + b_j = output - y
            #sum a_ij input_i - output + y = -b_j
            vars = np.concatenate([input_vars, [output_vars[i], y[i]]]) # np.append(x, [o[i], y[i]])
            rhs = [float(-self.b[i])]
            if self.op == 'FC':
                if (len(self.w.shape) == 1):
                    coefs = np.concatenate([[self.w[i]], [-1, 1]])
                else:
                    coefs = np.concatenate([self.w[:,i], [-1, 1]])
            elif self.op == 'Conv':
                coefs = np.concatenate([self.w[:,i], [-1, 1]])
            tag_con = tag + "_linear" + "_" + str(i) + "_" + str(output_idx)

            #print(coefs.shape)
            #print(vars.shape)
            ilpsolver.add_linear_constraint(rhs=rhs,
                                            senses='E',
                                            vars=vars,
                                            coefs=coefs,
                                            tag=tag_con)
            #z = 0 → output <= 0
            ind_vars = z[i]
            vars = [output_vars[i]]
            rhs = 0
            coefs = [1]
            tag_con = tag + "_ind_1_" + str(i) + "_" + str(output_idx)

            ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                               vars=vars,
                                               coefs=coefs,
                                               rhs=rhs,
                                               senses='L',
                                               complemented=1,
                                               tag=tag_con)
            #z = 1 → y <= 0
            ind_vars = z[i]
            vars = [y[i]]
            rhs = 0
            coefs = [1]
            tag_con = tag + "_ind_2_" + str(i) + "_" + str(output_idx)

            ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                               vars=vars,
                                               coefs=coefs,
                                               rhs=rhs,
                                               senses='L',
                                               complemented=0,
                                               tag=tag_con)
            # EXTRA for convinience z = 1 → output >= 0
            ind_vars = z[i]
            vars = [output_vars[i]]
            rhs = 0.00000005
            coefs = [1]
            tag_con = tag + "_ind_3_" + str(i) + "_" + str(output_idx)

            ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                               vars=vars,
                                               coefs=coefs,
                                               rhs=rhs,
                                               senses='G',
                                               complemented=0,
                                               tag=tag_con)

        # Constraints for discrete inputs
        if self.bin_input_names is not None:
            for i in range(len(self.possible_values)):
                #v = 1 → x = possible_values[i]
                ind_vars = self.bin_input_names[i]
                vars = [input_vars[0]]
                rhs = self.possible_values[i]
                coefs = [1]
                tag_con = tag + "_ind_discrete_" + str(i) + "_" + str(output_idx)

                ilpsolver.add_indicator_constraint(indicator_vars=ind_vars,
                                                   vars=vars,
                                                   coefs=coefs,
                                                   rhs=rhs,
                                                   senses='E',
                                                   complemented=0,
                                                   tag=tag_con)

            vars = np.array(self.bin_input_names)
            coefs = [1 for _ in vars]
            rhs = [1]
            tag_con = tag + "_plus_discrete_" + str(output_idx)

            ilpsolver.add_linear_constraint(rhs=rhs,
                                            senses='E',
                                            vars=vars,
                                            coefs=coefs,
                                            tag=tag_con)


def approximate_points(points):
    matrix = np.vstack(points)
    m_ = matrix[0]
    matrix = matrix - m_

    U, S, V = np.linalg.svd(matrix)

    if not (np.abs(S) < epsilon).any():
        dim_to_remove = S.shape[0]
    else:
        dim_to_remove = np.argmax(np.abs(S) < epsilon)

    diag_l = S.shape[0]
    reconstruct_diag = np.zeros((len(points), 128))
    reconstruct_diag[:diag_l,:diag_l] = np.diag(S)

    new_pts = np.matmul(U,reconstruct_diag)
    ld_points = new_pts[:, :dim_to_remove]

    def toHull(low_dim_pts_reduct):
        if low_dim_pts_reduct.shape[1] == 1:
            lb = np.min(low_dim_pts_reduct)
            ub = np.max(low_dim_pts_reduct)
            # x <= ub  -x >= -lb
            A = np.array([[1.0],[-1.0]])
            b = np.array([ub, -lb])
        else:
            # convex hull will get you Ax + b <= 0
            hull = ConvexHull(points=low_dim_pts_reduct)
            A = hull.equations[:,:-1]
            b = -hull.equations[:,-1]
        return A,b

    A,b = toHull(ld_points)

    # Ax<b
    n_rows = A.shape[0]
    n_cols = A.shape[1]

    num_var_to_add = 128 - (dim_to_remove)

    zero4 = np.zeros((n_rows,num_var_to_add))
    rows_to_append = np.zeros((2*num_var_to_add, n_cols+num_var_to_add))

    for idx in range(num_var_to_add):
        rows_to_append[idx*2,n_cols+ idx] = 1
        rows_to_append[idx*2+1,n_cols+ idx] = -1

    trueA = np.vstack([np.hstack([A, zero4]), rows_to_append])
    trueB = np.append(b, [0.0, 0.0]*num_var_to_add)

    finalA = np.matmul(trueA, V)
    finalB = np.matmul(finalA, m_) + trueB

    return finalA, finalB
