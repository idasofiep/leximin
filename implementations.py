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
from time import time

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

    grb.setParam("OutputFlag", 0)
    eps = 0.001

    model = Model()
    model.Params.TimeLimit = 120
    # model.setParam("Method", 2)
    A = model.addMVar((M,N), vtype=GRB.BINARY, lb=0., name="A") # Allocation matrix.
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
        model.optimize()
        if (model.status == GRB.TIME_LIMIT):
            return A, model
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

    grb.setParam("OutputFlag", 0)
    eps = 0.0001

    model = Model()
    model.Params.TimeLimit = 120
    # model.setParam("Method", 2)

    X = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=0., name="X") # panel probability vector
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
        model.optimize()
        if (model.status == GRB.TIME_LIMIT):
            return X, model
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
    model.Params.TimeLimit = 120
    A = model.addMVar((M,N), vtype=GRB.BINARY, lb=0., name="A") # Allocation matrix.
    h = model.addMVar((R,M), vtype=GRB.CONTINUOUS, lb=0., name="h") # h matrix

    funcs = [sum(instance[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) <= 1 + eps) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) >= 1 - eps) for j in range(N)) # N constraints, ensure all items must be allocated.
    model.addConstrs((h[k, j] >= -(funcs[j] - values_list[k])) for j in range(M) for k in range(R)) # M**2 constraints

    # solve as a lexicographic optimization problem with R-1 objectives (possible function values)
    for k in range(R-1):
        objective = sum(h[k+1,j] for j in range(M))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if (model.status == GRB.TIME_LIMIT):
            return A, model
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

    grb.setParam("OutputFlag", 0)
    R = 100 # number of panels to draw from in lottery, this value could also be set an input value
    eps = 0.0001

    model = Model()
    model.Params.TimeLimit = 120
    X = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=0., name="X") # Panel probabilities
    model.addConstr(sum(X[i] for i in range(M)) <= 1 + eps) # sum of probabilities should add up to one
    model.addConstr(sum(X[i] for i in range(M)) >= 1 - eps) # sum of probabilities should add up to one

    funcs = [sum(instance[j][i]*X[j] for j in range(M)) for i in range(N)] # there is one probability function for each person

    h = model.addMVar((R,N), vtype=GRB.CONTINUOUS, lb=0., name="h") # h matrix

    model.addConstrs((h[k, j] >= -(funcs[j] - (k/R))) for j in range(N) for k in range(R)) # N**2 constraints

    # solve as a lexicographic optimization problem with R-1 objectives (possible function values)
    for k in range(R-1):
        objective = sum(h[k+1,j] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if (model.status == GRB.TIME_LIMIT):
            return X, model
        z = model.objVal
        model.addConstr(objective <= z + eps)
    
    return X, model

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
        currently_fixed = fixed_agents
        model = Model()
        model.Params.TimeLimit = 120
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
        if (model.status == GRB.TIME_LIMIT):
            return A, model
        objectiveValue = model.objVal

        for i in range(M):
            if (fixed_binary[i] > 0.5):
                continue
            
            new_model = Model()
            new_model.Params.TimeLimit = 120
            new_A = new_model.addMVar((M,N), vtype=GRB.BINARY, name="new_A") # Allocation matrix.
            new_model.addConstrs((sum(new_A[k,j] for k in range(M)) <= 1 + eps) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
            new_model.addConstrs((sum(new_A[k,j] for k in range(M)) >= 1 - eps) for j in range(N)) # N constraints, ensure all items must be allocated
            funcs = [sum(instance[k][j]*new_A[k,j] for j in range(N)) for k in range(M)] # M value functions, one for each agent.
            
            for j in range(M):
                if (fixed_binary[j] < 0.5):
                    new_model.addConstr(funcs[j] >= objectiveValue - eps)
                else:
                    new_model.addConstr(funcs[j] >= fixed_values[j] - eps)
            
            new_model.setObjective(funcs[i], GRB.MAXIMIZE)
            new_model.optimize()
            if (new_model.status == GRB.TIME_LIMIT):
                return A, model
            new_objective_value = new_model.objVal

            if ((new_objective_value < objectiveValue + eps) & (new_objective_value > objectiveValue - eps)):
                # agent i's objective value can not be improved so the agent is now fixed
                fixed_values[i] = objectiveValue
                fixed_binary[i] = 1
                fixed_agents+=1
        if (currently_fixed == fixed_agents):
            return A, model
    return A, model

"""
SATURATION ALGORITHM FOR STRATIFICATION PROBLEMS

While there exists free objectives:

max z
s.t.  x in X,
      f_k(x) >= fixed_values[k]      for all saturated objectives k,
      f_k(x) >= z                    for all free objectives k

If the optimal f_j(x) value equals z, then try maximizing f_j(x).
If f_j(x) still equals z, then saturate objective j. 
otherwise, the optimal value for f_j must be larger than z; objective j remains free for now. 

This is a general saturation problem that works for a larger set of problems, but not all.
The method use a optimization model to decide wether objective j is saturated or not.
If the problem instance is feasible, at least one objective will be saturated for each iteration. 
"""
def saturation_stratification(instance):
    M = len(instance) # number of panels
    N = len(instance[0]) # number of people
    
    grb.setParam("OutputFlag", 0)
    eps = 0.0001

    fixed_people = 0 # counter for fixed agents
    fixed_values = [0]*N # fixed value for person i's function
    fixed_binary = [0]*N # indicate wether person i is fixed or not

    while (fixed_people < N):
        currently_fixed = fixed_people
        model = Model()
        model.Params.TimeLimit = 120
        X = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=0., name="X") # Panel probabilities
        model.addConstr(sum(X[i] for i in range(M)) <= 1 + eps) # sum of probabilities should add up to one
        model.addConstr(sum(X[i] for i in range(M)) >= 1 - eps) # sum of probabilities should add up to one

        funcs = [sum(instance[j][i]*X[j] for j in range(M)) for i in range(N)] # there is one probability function for each person

        z = model.addVar(vtype=GRB.CONTINUOUS, lb=0., name="z")

        for i in range(N):
            if (fixed_binary[i] < 0.5):
                model.addConstr(funcs[i] >= z)
            else:
                model.addConstr(funcs[i] >= fixed_values[i])

        model.setObjective(z, GRB.MAXIMIZE)
        model.optimize()
        if (model.status == GRB.TIME_LIMIT):
            return X, model
        objectiveValue = model.objVal #maxmin value

        for i in range(N):
            if (fixed_binary[i] > 0.5):
                continue
            
            new_model = Model() # new model to decide if agent i is saturated in this round or not
            new_model.Params.TimeLimit = 120
            new_X = new_model.addMVar(M, vtype=GRB.CONTINUOUS, lb=0., name="new_X") # Panel probabilities
            new_model.addConstr(sum(new_X[j] for j in range(M)) <= 1 + eps) # sum of probabilities should add up to one
            new_model.addConstr(sum(new_X[j] for j in range(M)) >= 1 - eps) # sum of probabilities should add up to one
            funcs = [sum(instance[j][k]*new_X[j] for j in range(M)) for k in range(N)] # there is one probability function for each person

            for j in range(N):
                if (fixed_binary[j] < 0.5):
                    new_model.addConstr(funcs[j] >= objectiveValue)  # instead of the variable z, 
                                                                     # lock each function to the maximized value of z 
                                                                     # found in the main optimization problem.
                else:
                    new_model.addConstr(funcs[j] >= fixed_values[j]) # also, lock all already saturated objectives
            
            new_model.setObjective(funcs[i], GRB.MAXIMIZE) # now, maximize the value function for agent i
            new_model.optimize()
            if (new_model.status == GRB.TIME_LIMIT):
                return X, model
            objective_value_new = new_model.objVal

            # if agent i can not improve its probability value, it is saturated.
            if ((objective_value_new < objectiveValue + eps) & (objective_value_new > objectiveValue - eps)):
                    fixed_values[i] = objectiveValue
                    fixed_binary[i] = 1
                    fixed_people+=1
        
        if (currently_fixed == fixed_people):
            return X, model
    return X, model
    