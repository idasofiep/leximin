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
        assert len(instance) != 0
        assert len(instance[0]) != 0
    except OSError as e:
        print(f"{type(e)}: {e}")
    return instance


"""
ORDERED OUTCOMES ALGORITHM FOR ALLOCATION PROBLEMS

Solve allocation problems with the Leximin Ordered Outcomes method described by Ogryczak and Sliwinśki in:
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)

Takes the problem instance matrix as input and
return an allocation matrix and the gurobi MIP model used to solve the problem. 

lex min [t_1 + sum(d_1j), ... , t_m + sum(d_mj)]

 s.t.   t_k + d_kj >= f_j(X)
        d_kj >= 0
 
"""
def ordered_outcomes_allocation(p):
    M = len(p) # number of agents
    N = len(p[0]) # number of items

    grb.setParam("OutputFlag", 0)

    eps = 0.0001

    model = Model()
    A = model.addMVar((M,N), vtype=GRB.BINARY, name="A") # Allocation matrix.
    t = model.addMVar(M, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length M, one for each agent. All real numbers. 
    d = model.addMVar((M,M), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length M x M. Only positive values.

    funcs = [sum(p[i][j]*A[i,j] for j in range(N)) for i in range(M)] # M value functions, one for each agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) <= 1 + eps) for j in range(N)) # N constraints, ensure no item can be allocated to more than one agent.
    model.addConstrs((sum(A[i,j] for i in range(M)) >= 1 - eps) for j in range(N)) # N constraints, ensure all items must be allocated. (not necessary for leximax, but for leximin it is)
    model.addConstrs((t[k] + d[k,j] >= funcs[j]) for j in range(M) for k in range(M)) # M**2 constraints

    # solve as a lexicographic optimization problem with M objectives. 
    for i in range(M):
        objective = (i+1) * t[i] + sum(d[i,j] for j in range(M))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        z = model.objVal
        model.addConstr(objective <= z + eps)
    
    return A, model



"""
Helper method to show results of allocations
"""
def print_allocation_result(allocation, instance):
    print("*********************")
    print("RESULT ALLOCATION USING THE ORDERED OUTCOMES METHOD")
    print("")
    print(allocation.X)
    print("")
        
    for i in range(len(instance)):
        assigned = []
        value = 0
        for j in range(len(instance[0])):
            if allocation[i][j].X > 0.5:
                assigned.append(j+1)
                value += instance[i][j]
        print("Agent " + str(i+1) + " is assigned items " + str(assigned) + " with a value sum of " + str(value))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("problemtype", type=str, help="problem type, either allocation or stratification")
    parser.add_argument("solvertype", type=str, help="solver algorithm, either oo (ordered outcomes), ov (ordered values), di (distribution) or so")
    parser.add_argument("filename", type=str, help="filename input (str)")
    args = parser.parse_args()
    filename = args.filename
    problemtype = args.problemtype
    solvertype = args.solvertype
    
    #ALLOCATIONS
    if (problemtype == "allocations"):

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo"):
            instance = read_csv("examples/allocations/" + filename)
            A, model = ordered_outcomes_allocation(instance)
            print_allocation_result(A, instance)
        

        
        

