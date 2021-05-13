import numpy as np


def block_solution(ilpsolver, sol, var_names, tag):
      coef = []
      b = 0
      nb_vars = len(var_names)
      for i in range(len(var_names)):
          if (sol[i] == 0):
              # relu s off
              coef.append(1)
          else:
              # on
              coef.append(-1)
              b += 1

      # x not y
      # not x or y
      # (1 - x) + y >= 1
      coef = coef
      rhs = -b + 1
      vars = var_names
      tag_con = "blocking" + "_" + str(tag)

      constraint_senses = ["G"]
      first_constraint = [vars, coef]
      rhs = [rhs]
      constraint_names = [tag_con]
      constraints = [first_constraint]

      c = ilpsolver.solver.linear_constraints.add(lin_expr=constraints,
                                                  senses=constraint_senses,
                                                  rhs=rhs,
                                                  names=constraint_names)
      return c[0]


def get_solution(ilpsolver, solution, var_names, sol_id=0):
    sol_vars = []
    sol_values = []
    for var_name in var_names:
        sol_vars.append(var_name)
        v = round(solution.get_values(sol_id, var_name), 5)
        sol_values.append(v)
    return sol_vars, sol_values


def enum_solutions(ilpsolver, block_vars_names, vars_print, stop_cond=None, stop_after_one=False, verbose=True):
    sol_values_total = list()
    while(True):
        # Uncomment for fast debugging
        #if len(sol_values_total) > 100:
        #    if verbose: print('Max number of solutions reached')
        #    break
        ilpsolver.solver.parameters.mip.pool.intensity.set(1)
        #ilpsolver.solver.parameters.emphasis.numerical.set(1)
        #ilpsolver.solver.parameters.mip.tolerances.integrality.set(0.00001)
        if stop_after_one:
            ilpsolver.solver.parameters.mip.limits.populate.set(1)

        ilpsolver.solver.populate_solution_pool()
        if (ilpsolver.solver.solution.get_status() == ilpsolver.solver.solution.status.MIP_infeasible):
            if verbose: print("UNSAT")
            break
        if (ilpsolver.solver.solution.is_primal_feasible()):
            solutions = ilpsolver.solver.solution.pool
        else:
            if verbose: print("UNFEAS")
            break

        nb_solutions = ilpsolver.solver.solution.pool.get_num()
        if verbose: print( "Block {}, round {}  the solution pool contains {} solutions.".format("", round, nb_solutions))
        sol_values_all = list()
        for s in range(nb_solutions):
            vars, values = get_solution(ilpsolver, solutions, block_vars_names, sol_id=s)
            if (values in sol_values_total):
                continue

            sol_values_all.append(values)
            sol_values_total.append(values)

            if stop_after_one:
                return sol_values_total
            if stop_cond is not None and stop_cond(vars, values):
                return sol_values_total

            p_vars, p_values = get_solution(ilpsolver, solutions, vars_print, sol_id=s)
            for i, _ in enumerate(p_vars):
                print(p_vars[i], p_values[i])

        for s in range(len(sol_values_all)):
            c = block_solution(ilpsolver, sol_values_all[s], block_vars_names,  "_one_" + str(round) + "_" + str(s))
    if verbose: print("Total unique solutions {}".format(len(sol_values_total)))

    return sol_values_total

