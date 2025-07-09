"""
This program contains implementations of lexicographic max-min optimization algorithms,
for solving:
    - allocation problems of indivisible goods
    - stratification problems for producing uniform lotteries of citizens' assembly panels
    - stratification problems for producing non-uniform lotteries of citizens' assembly panels

The two leximin formulas that are used for all implementations are presented and described by
Ogryczak and Sliwinśki in:
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)

The citizens' assembly stratification problems and a different leximin solution framework is presented by
Flanigan et al. in:
"Fair Algorithms for selecting Citizens' Assemblies" (2021),
https://doi.org/10.1038/s41586-021-03788-6

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
import math


EPS = 0.0005
SOLVER_MIPGAP = 0.005


"""
ORDERED OUTCOMES ALGORITHMS
The Leximin Ordered Outcomes method was described by Ogryczak and Sliwinśki in:
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)
"""

"""
ORDERED OUTCOMES METHOD FOR SOLVING ALLOCATION PROBLEMS
This method solve allocation problems where some undivisible goods or resources
are to be shared among a set of people, so that each persons utility is maximized
according to the leximin principle. 

Input:
- The problem instance matrix, a M x N matrix,
  where each row is a persons' values for the items to be shared. 
- A solver time limit

Return:
- An binary allocation matrix of size M x N, (saying which items are allocated to each person)
- The gurobi MIP model used to solve the problem. 
 
"""
def ordered_outcomes_allocation(instance, time_limit):
    M = len(instance) # number of agents
    N = len(instance[0]) # number of items

    grb.setParam("OutputFlag", 0)

    model = Model()

    model.Params.TimeLimit = time_limit
    # model.Params.MIPGap = SOLVER_MIPGAP

    A = model.addMVar((M,N), vtype=GRB.BINARY, lb=0., name="A") # Allocation binary matrix.
    t = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length M, one for each agent. All real numbers. 
    d = model.addMVar((M,M), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length M x M. Only positive values.

    # each person has a corresponding probability function, calculating the number of panels that person is part of
    funcs = [sum(instance[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent. 
    model.addConstrs((sum(A[i,j] for i in range(M)) == 1) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
    
    # constraints as defined by ordered outcomes formula.
    model.addConstrs((t[k] + d[k,j] >= - funcs[j]) for j in range(M) for k in range(M)) # M**2 constraints

    start = time()
    # solve as a lexicographic optimization problem with M objectives. 
    for i in range(M):
        objective = (i+1) * t[i] + sum(d[i,j] for j in range(M))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if ((time()-start) > time_limit): # time limit is reached
            return A, model
        try:
            z = model.objVal
        except Exception as e:
            print("An exception occured trying to retrieve objective value. ")
            return A, model
        model.addConstr(objective <= z + EPS)
    return A, model

"""
ORDERED OUTCOMES ALGORITHM FOR STRATIFICATION PROBLEMS (UNIFORM LOTTERY)
This method produce a uniform lottery of panels, 
so that each persons probability of being selected by the lottery
is optimized according to the leximin principle.

Input:
- People pool (binary matrix), each row corresponds to a person and their membership status for each category,
- Panel_size value (number of people in the panel)
- Lottery_size value (number of panels in lottery)
- Quotas (minimal number of people in each category that should be in the panel)
- Solver time limit

Return:
- A matrix of size (lottery_size x panel_size) that is the uniform lottery of panels
- The gurobi optimization model

"""
def ordered_outcomes_stratification_uniform(people, panel_size, lottery_size, quotas, time_limit):
    M = len(people) # number of people
    N = len(quotas) # number of categories

    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    # model.Params.MIPGap = SOLVER_MIPGAP

    panels = [model.addMVar(M, vtype=GRB.BINARY) for i in range(lottery_size)] 
    # each panel have binary values for every person, so committees[i][j] = 1 means that person j is member of panel i.
    
    # set quota restrictions for all panels
    for panel in panels:
        for i in range(N):
            model.addConstr(grb.quicksum(people[j][i]*panel[j] for j in range(M)) >= quotas[i][0]) # lower qouta
            model.addConstr(grb.quicksum(people[j][i]*panel[j] for j in range(M)) <= quotas[i][1]) # upper quota
        model.addConstr(grb.quicksum(panel[i] for i in range(M)) == panel_size) # also, each panel must have excactly panel_size number of people as members

    t = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length M, one for each person. All real numbers. 
    d = model.addMVar((M,M), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of size M x M. Only positive values.

    # each person has a corresponding probability function, calculating the number of panels that person is part of
    funcs = [grb.quicksum(panel[i] for panel in panels) for i in range(M)]

    # constraints as defined by ordered outcomes formula.
    model.addConstrs((t[k] + d[k,i] >= - funcs[i]) for i in range(M) for k in range(M)) # N**2 constraints

    start = time()
    # solve as a lexicographic optimization problem with M objectives
    for i in range(M):
        objective = (i+1) * t[i] + sum(d[i,j] for j in range(M)) # objective as defined by ordered outcomes method
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        if ((time()-start) > time_limit): # reached solver time limit
            return panels, model
        try:
            z = model.objVal
        except Exception as e:
            print("An exeption occured when trying to retrieve the objective value.")
            return panels, model
        # add the new objective value as a constraint and continue
        model.addConstr(objective <= z + EPS)
    
    return panels, model

"""
ORDERED OUTCOMES ALGORITHM FOR STRATIFICATION PROBLEMS (NON-UNIFORM / GENERAL LOTTERIES)
This method produce a non-uniform/general lottery of panels, 
so that each persons probability of being selected by the lottery
is optimized according to the leximin principle.

In contrast to the uniform method, 
this method does not produce a lottery where each panel is selected with the same probability.
Instead it produce excactly (N+1) panels, where N is the number of people,
and each panel has a corresponding continous variable for that panels likelihood of being selected
in the lottery.

The method takes advantage of the Carathéodory's theorem, which states 
that there is at most N+1 unique panels in the optimal solution.
This is described by Flanigan et al. in chapter 7 of their supplementary work of
"Fair Algorithms for selecting Citizens' Assemblies" (2021),
https://doi.org/10.1038/s41586-021-03788-6

Input:
- People pool (binary matrix), each row corresponds to a person and their membership status for each category,
- Panel_size value (number of people in the panel)
- Quotas (minimal number of people in each category that should be in the panel)
- Solver time limit

Return:

- The panel probability values
- The panels in the lottery
- The gurobi optimization model

"""
def ordered_outcomes_stratification_general(people, panel_size, quotas, time_limit):
    M = len(people) # number of people
    N = len(quotas) # number of categories
    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    # model.Params.MIPGap = SOLVER_MIPGAP

    panels = [model.addMVar(M, vtype=GRB.BINARY) for i in range(M+1)] # each panel have binary values for every person
    panel_vars = model.addMVar((M+1),vtype=GRB.CONTINUOUS, lb=0.) #continous and non-negative variables, this is the probability values for each panel

    # ensure quotas are respected for all panels
    for panel in panels:
        for i in range(N): # for all categories
            model.addConstr(grb.quicksum(people[j][i]*panel[j] for j in range(M)) >= quotas[i][0]) # lower quota
            model.addConstr(grb.quicksum(people[j][i]*panel[j] for j in range(M)) <= quotas[i][1]) # upper quota
        model.addConstr(grb.quicksum(panel[i] for i in range(M)) == panel_size) # ensure excactly panel_size number of people in each panel
    
    model.addConstr(grb.quicksum(panel_vars[i] for i in range(M+1)) == 1) # panel probabilities should add up to 1. 

    # additional variables as defined by the ordered outcomes method
    t = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length M, one for each person. All real numbers. 
    d = model.addMVar((M,M), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length M x M. Only positive values.

    # each person has a corresponding function calculating their probability of being selected in the lottery.
    # The function does this by, for each person j, sum all panel probabilities for panels where person j is member.
    funcs = [grb.quicksum(panels[i][j]*panel_vars[i] for i in range(M+1)) for j in range(M)]

    # constraints as defined by ordered outcomes method
    model.addConstrs((t[k] + d[k,i] >= - funcs[i]) for i in range(M) for k in range(M)) # M**2 constraints

    start = time()
    # solve as a lexicographic optimization problem with M objectives
    for i in range(M):
        objective = (i+1) * t[i] + sum(d[i,j] for j in range(M)) # as defined by ordered outcomes method
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if ((time()-start) > time_limit):
            return panel_vars, panels, model
        try:
            z = model.objVal
        except Exception as e:
            print("An exception occured when trying to retrieve the objective value. ")
            return panel_vars, panels, model
        
        # add objective and objective value as a new constraint and continue
        model.addConstr(objective <= z + EPS)
    
    return panel_vars, panels, model

"""
ORDERED VALUES ALGORITHM

The Leximin Ordered Values method described by Ogryczak and Sliwinśki in:
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)

lex min [sum(h_2j), ... , sum(h_rj)]
 s.t.     h_kj >= f_j(X) - v_k
          h_kj >= 0

An important difference from the ordered outcomes method is that we need to compute 
or approximate all possible function values to use this method.
Note that the algorithm works also if there are values in the list that are not part of the final solution, 
so by approximation we can make a list that _at least_ has all possible values, 
but might have values that are not part of this set; the method will still produce the same result.
"""

"""
ORDERED VALUES METHOD FOR ALLOCATION PROBLEMS

Input:
- the problem instance matrix
- a list of function values
- a solver time limit

Return:
- Return an allocation matrix
- The gurobi MIP model used to solve the problem
 
"""
def ordered_values_allocation(instance, values_list, time_limit):
    M = len(instance) # number of agents
    N = len(instance[0]) # number of items
    R = len(values_list) # number of possible values

    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    # model.Params.MIPGap = SOLVER_MIPGAP

    A = model.addMVar((M,N), vtype=GRB.BINARY, lb=0., name="A") # Allocation matrix.
    h = model.addMVar((R,M), vtype=GRB.CONTINUOUS, lb=0., name="h") # h matrix, as defined by ordered values method

    funcs = [sum(instance[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) == 1) for j in range(N)) # ensure no item can be allocated to more than one agent.

    # constraints as defined by ordered values method
    model.addConstrs((h[k, j] >= -(funcs[j] - values_list[k])) for j in range(M) for k in range(R)) # M*R constraints

    start = time()
    # solve as a lexicographic optimization problem with R-1 objectives (possible function values)
    for k in range(R-1):
        objective = sum(h[k+1,j] for j in range(M)) # as defined in ordered values method
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        if ((time()-start) > time_limit): # time limit reached
            return A, model
        
        try:
            z = model.objVal
        except Exception as e:
            print("An exception occured when trying to retrieve objective value. ")
            return A, model
        
        # add objective and objective value as constraint and continue
        model.addConstr(objective <= z + EPS)

    return A, model

"""
ORDERED VALUES ALGORITHM FOR STRATIFICATION PROBLEMS (UNIFORM LOTTERIES)
The method produce a uniform lottery that leximin maximized each persons probability of being selected.

In contrast to the allocation problems, we don't need a list of possible function values,
as we know that each persons probabilities in a uniform lottery is 
the number of panels they are member of divided by the total number of panels. 
So if we want to make a lottery with 1000 panels, the possible function/probability values are all
thousandths from 0 and 1.

However, the algorithm runtime can be improved by instead providing a reduced value list
of possible function values, possible produced by a heuristic function. 

Input:
- A list of people, each person is described as a binary list for membership status in each category.
- Panel_size, how many members will each panel have
- Lottery size, how many panels do we want in the lottery
- quotas, a list of upper and lower quotas for each category
- solver time limit

Return:
- the panels that form the uniform lottery
- the optimization model
 
"""

def ordered_values_stratification_uniform(people, panel_size, lottery_size, quotas, time_limit):
    M = len(people) # number of people
    N = len(quotas) # number of quotas

    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    # model.Params.MIPGAP = SOLVER_MIPGAP

    #each panel have binary values for every person
    panels = [model.addMVar(M, vtype=GRB.BINARY) for i in range(lottery_size)] 
    
    # ensure that quotas are respected
    for panel in panels:
        for i in range(N):
            model.addConstr(grb.quicksum(people[j][i]*panel[j] for j in range(M)) >= quotas[i][0]) # lower quota
            model.addConstr(grb.quicksum(people[j][i]*panel[j] for j in range(M)) <= quotas[i][1]) # upper quota
        model.addConstr(grb.quicksum(panel[i] for i in range(M)) == panel_size) # excactly panel_size number of people in each committee
    
    h = model.addMVar((lottery_size,M), vtype=GRB.CONTINUOUS, lb=0., name="h") # h matrix

    # function values for each person
    funcs = [grb.quicksum((panel[i]/lottery_size) for panel in panels) for i in range(M)]

    # instead of using a list of function values, we use all fractions of 1/lottery_size from 0 to 1.
    model.addConstrs((h[k, j] >= -(funcs[j] - (k/lottery_size))) for j in range(M) for k in range(lottery_size))

    start = time()
    # solve as a lexicographic optimization problem with one objective for each committee (possible function values)
    for k in range(lottery_size-1):
        objective = sum(h[k+1,j] for j in range(M))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if ((time() - start) > time_limit): # time limit reached
            return panels, model
        try:
            z = model.objVal
        except Exception as e:
            print("An exception occured when trying to retrieve the objective value. ")
            return panels, model
        model.addConstr(objective <= z + EPS)
    return panels, model


