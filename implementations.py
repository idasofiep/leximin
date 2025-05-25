"""
This program has implementations of different lexicographic max-min optimization formulas.
Two types of problems can be solved; allocation problems and sortition problems.
"""

import gurobipy as grb
from gurobipy import Model, GRB
import mip
import csv
import numpy as np
import typing
from typing import NewType, List
from dataclasses import dataclass
from argparse import ArgumentParser, RawDescriptionHelpFormatter

"""
Read a csv file that define the problem instance
 - Allocation problems are defined as integer matrices
 - Stratification problems are defined as binary matrices

return M x N integer matrix
"""
def read_csv(filename):
    instance = []
    try:
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=" ")
            for line in csv_reader:
                values = []
                for val in line:
                    values.append(int(val))
                instance.append(values)
    except OSError as e:
        print(f"{type(e)}: {e}")
    return instance


"""
ORDERED OUTCOMES ALGORITHM FOR ALLOCATION PROBLEMS

Solve allocation problems with the Leximin Ordered Outcomes method described by Ogryczak and Sliwinśki in:
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)

- Takes the problem instance matrix as input and
- Return an allocation matrix and the gurobi MIP model used to solve the problem. 

lex min [t_1 + sum(d_1j), ... , t_m + sum(d_mj)]

 s.t.   t_k + d_kj >= f_j(X)
        d_kj >= 0
 
"""
def ordered_outcomes_allocation(instance):
    M = len(instance) # number of agents
    N = len(instance[0]) # number of items

    grb.setParam("OutputFlag", 1)
    eps = 0.0001

    model = Model()
    # model.setParam("Method", 2)
    A = model.addMVar((M,N), vtype=GRB.BINARY, name="A") # Allocation matrix.
    t = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length M, one for each agent. All real numbers. 
    d = model.addMVar((M,M), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length M x M. Only positive values.

    funcs = [sum(instance[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) <= 1 + eps) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) >= 1 - eps) for j in range(N)) # N constraints, ensure all items must be allocated. (not necessary for leximax, but for leximin it is)
    model.addConstrs((t[k] + d[k,j] >= - funcs[j]) for j in range(M) for k in range(M)) # M**2 constraints

    # solve as a lexicographic optimization problem with M objectives. 
    for i in range(M):
        objective = (i+1) * t[i] + sum(d[i,j] for j in range(M))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize() # TODO: add handling for infeasible model?
        # NOTE: for large problem instances, this optimization takes a lot of time,
        # but if the optimization is interrupted with "Ctrl + C" after a short while
        # for every iteration, it produces the correct solution in significantly shorter time
        # there is something about number of nodes or iterations that are not optimal...
        z = model.objVal
        model.addConstr(objective <= z + eps)
    
    return A, model

"""
ORDERED OUTCOMES ALGORITHM FOR STRATIFICATION PROBLEMS


- Takes the problem instance matrix as input and
- Return an allocation matrix and the gurobi MIP model used to solve the problem. 

lex min [t_1 + sum(d_1j), ... , t_m + sum(d_mj)]

 s.t.   t_k + d_kj >= f_j(X)
        d_kj >= 0

In contrast to the allocation oo method, 
the functions to be lexicographically min-maxxed 
is each panel's probability of being chosen in a lottery.

"""
def ordered_outcomes_stratification(instance):
    M = len(instance) # number of panels
    N = len(instance[0]) # number of people

    grb.setParam("OutputFlag", 1)
    eps = 0.0001

    model = Model()
    # model.setParam("Method", 2)

    X = model.addMVar(M, vtype=GRB.CONTINUOUS, name="X") # panel probability vector
    model.addConstr(sum(X[i] for i in range(M)) <= 1 + eps) # sum of probabilities should add up to one
    model.addConstr(sum(X[i] for i in range(M)) >= 1 - eps) # sum of probabilities should add up to one

    funcs = [sum(instance[j][i]*X[j] for j in range(M)) for i in range(N)]

    t = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length M, one for each agent. All real numbers. 
    d = model.addMVar((N,N), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length M x M. Only positive values.

    model.addConstrs((t[k] + d[k,j] >= - funcs[j]) for j in range(N) for k in range(N)) # M**2 constraints

    # solve as a lexicographic optimization problem with N objectives. 
    for i in range(N):
        objective = (i+1) * t[i] + sum(d[i,j] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize() # TODO: add handling for infeasible model?
        # NOTE: for large problem instances, this optimization takes a lot of time,
        # but if the optimization is interrupted with "Ctrl + C" after a short while
        # for every iteration, it produces the correct solution in significantly shorter time
        # there is something about number of nodes or iterations that are not optimal...
        z = model.objVal
        model.addConstr(objective <= z + eps)
    
    return X, model

"""
ORDERED VALUES ALGORITHM FOR ALLOCATION PROBLEMS 

Solve allocation problems with the Leximin Ordered Values method described by Ogryczak and Sliwinśki in:
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)

An important difference from the ordered outcomes method is that we need to compute or approximate all possible function values to use this method.
Note that the algorithm works also if there are values in the list that are not part of the final solution, so by approximation
we can make a list that _at least_ has all possible values, but might have values that are not part of this set; the method will still produce the same result.
How we can find these values depend on the problem instance, but for all integer problem types all we need is a lower and upper bound,
and then we can run the algorithm with all integer values between these two values. The value list is found in a separate method (create_integer_func_values).

- Takes the problem instance matrix and the function values list as input
- Return an allocation matrix and the gurobi MIP model used to solve the problem

lex min [sum(h_2j), ... , sum(h_rj)]
 s.t.     h_kj >= f_j(X) - v_k
          h_kj >= 0
 
"""
def ordered_values_allocation(instance, values_list):
    M = len(instance) # number of agents
    N = len(instance[0]) # number of items

    grb.setParam("OutputFlag", 0)
    R = len(values_list)

    eps = 0.0001

    model = Model()
    A = model.addMVar((M,N), vtype=GRB.BINARY, name="A") # Allocation matrix.
    h = model.addMVar((R,M), vtype=GRB.CONTINUOUS, name="h") # h matrix

    funcs = [sum(instance[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) <= 1 + eps) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) >= 1 - eps) for j in range(N)) # N constraints, ensure all items must be allocated.
    model.addConstrs((h[k, j] >= -(funcs[j] - values_list[k])) for j in range(M) for k in range(R)) # M**2 constraints

    # solve as a lexicographic optimization problem with R-1 objectives (possible function values)
    for k in range(R-1):
        objective = sum(h[k+1,j] for j in range(M))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        z = model.objVal
        model.addConstr(objective <= z + eps)
    
    return A, model

"""
ORDERED VALUES ALGORITHM FOR STRATIFICATION PROBLEMS

- Takes the problem instance matrix as input
- Return a list of probability values for each panel, and the gurobi MIP model

In contrast to the allocation problems, we don't need to find possible function values,
as we know that the function values are the possible probability values for each person,
which depend on the size of the lottery (or like the number of panels we want to draw from).
So if we want to make a lottery with 1000 panels, the possible function/probability values are all
thousandths between 0 and 1.

lex min [sum(h_2j), ... , sum(h_rj)]
 s.t.     h_kj >= f_j(X) - v_k
          h_kj >= 0
 
"""
def ordered_values_stratification(instance):
    M = len(instance) # number of panels
    N = len(instance[0]) # number of people

    grb.setParam("OutputFlag", 1)
    values_list = []
    R = 1000 # number of panels to draw from in lottery, this value could also be set an input value

    for i in range(R):
        values_list.append(i/R) # add all possible probability values to value list

    eps = 0.0001

    model = Model()
    X = model.addMVar(M, vtype=GRB.BINARY, name="X") # Panel probabilities
    model.addConstr(sum(X[i] for i in range(M)) <= 1 + eps) # sum of probabilities should add up to one
    model.addConstr(sum(X[i] for i in range(M)) >= 1 - eps) # sum of probabilities should add up to one

    funcs = [sum(instance[j][i]*X[j] for j in range(M)) for i in range(N)] # there is one probability function for each person

    h = model.addMVar((R,N), vtype=GRB.CONTINUOUS, name="h") # h matrix

    model.addConstrs((h[k, j] >= -(funcs[j] - values_list[k])) for j in range(N) for k in range(R)) # N**2 constraints

    # solve as a lexicographic optimization problem with R-1 objectives (possible function values)
    for k in range(R-1):
        objective = sum(h[k+1,j] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        z = model.objVal
        model.addConstr(objective <= z + eps)
    
    return X, model

"""
Helper method to create integer function value list

- Takes the problem instance as input
- Return a list of integers between 0 and the max value for the problem instance

TODO: Make support for other func_values than integers
"""
def create_integer_func_values(instance):
    max_value = 0
    for func in instance:
        max_value = max(max_value, sum(func[i] for i in range(len(func))))
    value_list = []
    for i in range(max_value+1):
        value_list.append(i)
    return value_list



"""
SATURATION ALGORITHM FOR ALLOCATION PROBLEMS

While there exists free objectives:

max z
s.t.  x in X,
      f_k(x) >= fixed_values[k]      for all saturated objectives k,
      f_k(x) >= z                    for all free objectives k

If the optimal f_j(x) value equals z, then objective j becomes saturated from now on.
otherwise, the optimal value must be larger than z; objective j remains free for now. 

"""
def saturation_allocation(instance):
    M = len(instance) # number of agents
    N = len(instance[0]) # number of items
    
    grb.setParam("OutputFlag", 0)
    eps = 0.0001

    fixed_agents = 0 # counter for fixed agents
    fixed_values = [0]*M # fixed value for agent i's function
    fixed_binary = [0]*M # indicate wether agent i is fixed or not

    
    while (fixed_agents < M):
        model = Model()
        A = model.addMVar((M,N), vtype=GRB.BINARY, name="A") # Allocation matrix.
        model.addConstrs((sum(A[i,j] for i in range(M)) <= 1 + eps) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
        model.addConstrs((sum(A[i,j] for i in range(M)) >= 1 - eps) for j in range(N)) # N constraints, ensure all items must be allocated

        funcs = [sum(instance[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent.

        z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
        for i in range(M):
            if (fixed_binary[i] < 0.5):
                model.addConstr(funcs[i] >= z)
            else:
                model.addConstr(funcs[i] >= fixed_values[i])

        model.setObjective(z, GRB.MAXIMIZE)
        model.optimize()
        objectiveValue = model.objVal

        for i in range(M):
            if (fixed_binary[i] < 0.5):
                func_value = sum(instance[i][j]*A[i,j].x for j in range(N)) # agent i's objective function value for this allocation
                if ((func_value < objectiveValue + eps) & (func_value > objectiveValue - eps)):
                    # agent i's objective value can not be improved so the agent is now fixed
                    fixed_values[i] = objectiveValue
                    fixed_binary[i] = 1
                    fixed_agents+=1
                    break
    
    return A, model

"""
Helper method to get sorted allocation values list
"""
def get_allocation_values(allocation, instance):
    values = []
    for i in range(len(instance)):
        assigned = []
        value = 0
        for j in range(len(instance[0])):
            if allocation[i][j].X > 0.5:
                assigned.append(j+1)
                value += instance[i][j]
        values.append(value)
    values.sort()
    return values


"""
Helper method to show results of allocations
"""
def print_allocation_result(allocation, instance):
    print("*********************")
    print("RESULT ALLOCATION PROBLEM: ")
    print("")
    print(allocation.X)
    print("")

    values = get_allocation_values(allocation, instance)

    print("LEXIMIN SORTED VALUES: ")
    print(values)

"""
Helper method for finding each person's probability of being chosen
"""

def get_stratification_probabilities(X, instance):
    people_probabilities = [0]*len(instance[0])
    for i in range(len(instance)):
        for j in range(len(instance[0])):
            people_probabilities[j] += X.x[i] * instance[i][j]
    return people_probabilities

"""
Helper method for printing stratification results
"""

def print_stratification_result(X, instance):
    print("*********************")
    print("RESULT STRATIFICATION PROBLEM: ")
    print("")
    print("Number of possible panels: " + str(len(instance)))
    print("")
    print("Each panel's probability value: ")
    print(X.x)
    print("")
    print("Number of people: " + str(len(instance[0])))
    print("")
    print("List of each persons probability: ")
    print(get_stratification_probabilities(X, instance))
    print("")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("problemtype", type=str, help="problem type, either allocation or stratification")
    parser.add_argument("solvertype", type=str, help="solver algorithm, either oo (ordered outcomes), ov (ordered values), sat (saturation)")
    parser.add_argument("filename", type=str, help="filename input (str)")
    args = parser.parse_args()
    filename = args.filename
    problemtype = args.problemtype
    solvertype = args.solvertype
    
    #ALLOCATIONS
    if (problemtype == "allocations"):
        instance = read_csv("examples/allocations/" + filename)

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo"):
            A, model = ordered_outcomes_allocation(instance)
        
        #ORDERED VALUES METHOD
        elif (solvertype == "ov"):
            values_list = create_integer_func_values(instance)
            A, model = ordered_values_allocation(instance, values_list)
        
        #SATURATION METHOD
        elif (solvertype == "sat"):
            A, model = saturation_allocation(instance)
        
        print_allocation_result(A, instance)
    
    #STRATIFICATIONS
    if (problemtype == "stratifications"):
        instance = read_csv("examples/stratifications/" + filename)

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo"):
            X, model = ordered_outcomes_stratification(instance)
        
        #ORDERED VALUES METHOD
        elif (solvertype == "ov"):
            X, model = ordered_outcomes_stratification(instance)
        
        print_stratification_result(X, instance)
        




        
        

