"""
In this program, we demonstrate how our implementations can be modified to make citizens' assembly lotteries.

The program contains a modification of the stratification.py program implemented by
Brett Hennig bsh [AT] sortitionfoundation.org and Paul Gölz goelz (AT) seas.harvard.edu,
their methods are implemented for comparison purposes. 

The original program can be found at
https://github.com/pgoelz/citizensassemblies-replication/blob/master/stratification.py
and their framework is described in:
"Fair Algorithms for selecting Citizens' Assemblies" (2021)
https://doi.org/10.1038/s41586-021-03788-6

Additionally, we have implemented methods for solving the citizens' assembly lottery problems of the same format,
using the leximin ordered outcomes and ordered values formulas presented by Ogryczak and Sliwinśki in
"On Direct Methods for Lexicographic Min-Max Optimization" (2006)
(https://doi.org/10.1007/11751595_85)
"""

import gurobipy as grb
from gurobipy import Model, GRB
import mip
import csv
import numpy as np
import typing
import random
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable, Optional, Set
from dataclasses import dataclass
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from copy import deepcopy
from itertools import combinations
from time import time
from analysis import plot_fairness_statistics

"""
Start of
Modification of https://github.com/pgoelz/citizensassemblies-replication/blob/master/stratification.py
"""
# class for throwing error/fail exceptions
class SelectionError(Exception):
    def __init__(self, message):
        self.msg = message

# Takes two file names as input, one for categories and one for the people pool. 
# Return two dictionaries, one for categories and one for the people pool.
# 
# categories: dict[str, dict[str, dict[str, int]]]             e.g. "gender": {"male": {"min": 4, "max": 6}, "female": {"min": 4, "max": 6}}
# people: dict[str, dict[str, str]]                            e.g  "13": {gender: female, "leaning": "conservative", "age": "over"}
# 
# Needs to be in a spesific format, like the files in the example_citizens_assembly folder.
# This format is accepted by panelot.org and can be used in the original stratification library.
# 
# This method does not handle errors. TODO: rewrite and handle exceptions
def read_csv(categories_file, peoples_file):
    categories = {}
    people = {}

    with open(categories_file, "r", newline="") as csvfile:
        category_reader = csv.DictReader(csvfile, delimiter=",")
        for row in category_reader:
            if row["category"] in categories.keys():
                categories[row["category"]].update({row["feature"]: {"min": int(row["min"]), "max": int(row["max"])}})
            else:
                categories.update({row["category"]: {row["feature"]: {"min": int(row["min"]), "max": int(row["max"])}}})
    
    with open(peoples_file, "r", newline="") as csvfile:
        peoples_reader = csv.DictReader(csvfile, delimiter=",")
        i=1
        for row in peoples_reader:
            people.update({i: {category: row[category] for category in categories.keys()}})
            i = i+1

    return categories, people

# makes a frozen set of the id's of the people chosen for this committee
def _ilp_results_to_committee(variables: Dict[str, mip.entities.Var]) -> FrozenSet[str]:
    try:
        res = frozenset(id for id in variables if variables[id].x > 0.5)
    except Exception as e:  # unfortunately, MIP sometimes throws generic Exceptions rather than a subclass.
        raise ValueError(f"It seems like some variables does not have a value. Original exception: {e}.")
    return res

# This method takes the categories, people pool and panel_size as input,
# and produce a set of initial panels. 
def _committee_generation(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]], panel_size: int):
    model = Model()

    # binary variables indicating wether the agent is in the committee or not
    agent_vars = {id: model.addVar(vtype=GRB.BINARY) for id in people}

    # for each committee, there should be panel_size number of agents chosen. 
    model.addConstr(sum(agent_vars.values()) == panel_size)

    # we have to respect quotas
    for feature in categories:
        for value in categories[feature]:
            number_feature_value_agents = sum(agent_vars[id] for id, person in people.items()
                                                   if person[feature] == value)
            model.addConstr(number_feature_value_agents >= categories[feature][value]["min"])
            model.addConstr(number_feature_value_agents <= categories[feature][value]["max"])
    
    # GENERATE INITIAL COMMITTEES
    committees: Set[FrozenSet[str]] = set() # set of committees discovered so far
    covered_agents: Set[str] = set() # all agents included in some committee

    # Each agent has a weight value that are reduced when an agent is chosen for a committee.
    # This is to ensure a variation in the initial commitees. Agents with larger weights are prioritized. 
    weights = {id: random.uniform(0.99, 1.0) for id in agent_vars}

    for i in range(3*len(people)):
        objective = sum(weights[id] * agent_vars[id] for id in agent_vars)
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        
        # new_set is a potential new panel to add in the initial lottery
        new_set = _ilp_results_to_committee(agent_vars)
        
        # all agents in the new_set has reduced their weigths.
        for id in new_set:
            weights[id] *= 0.8
        
        # We rescale the weights, which does not change the conceptual algorithm but prevents floating point problems.
        coefficient_sum = sum(weights.values())
        for id in agent_vars:
            weights[id] *= len(agent_vars) / coefficient_sum

        if new_set not in committees:
            committees.add(new_set)
            for id in new_set:
                covered_agents.add(id)
        else:
            # If our committee is already known, make all weights a bit more equal again to mix things up a little.
            for id in agent_vars:
                weights[id] = 0.9 * weights[id] + 0.1
        
    print(
        f"Multiplicative weights phase over, did {len(people)*3} rounds. Discovered {len(committees)}"
        " committees.")
    
    # If there are any agents that have not been included so far, try to find a committee including this specific agent.
    for id in agent_vars:
        if id not in covered_agents:
            new_committee_model.objective = agent_vars[id]  # only care about agent `id` being included.
            new_committee_model.optimize()
            new_set: FrozenSet[str] = _ilp_results_to_committee(agent_vars)
            if id in new_set:
                committees.add(new_set)
                for id2 in new_set:
                    covered_agents.add(id2)
            else:
                print(f"Agent {id} not contained in any feasible committee.")
    
    # We assume in this stage that the quotas are feasible.
    assert len(committees) >= 1

    if len(covered_agents) == len(agent_vars):
        print("All agents are contained in some feasible committee.")

    return model, agent_vars, committees, frozenset(covered_agents)


def _dual_leximin_stage(people: Dict[str, Dict[str, str]], committees: Set[FrozenSet[str]],
                        fixed_probabilities: Dict[str, float]):

    model = Model()
    agent_vars = {person: model.addVar(vtype=GRB.CONTINUOUS, lb=0.) for person in people}  # yᵢ
    cap_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.)  # ŷ
    model.addConstr(grb.quicksum(agent_vars[person] for person in people if person not in fixed_probabilities) == 1)
    for committee in committees:
        model.addConstr(grb.quicksum(agent_vars[person] for person in committee) <= cap_var)
    model.setObjective(cap_var - grb.quicksum(
        fixed_probabilities[person] * agent_vars[person] for person in fixed_probabilities),
                       GRB.MINIMIZE)

    # Change Gurobi configuration to encourage strictly complementary (“inner”) solutions. These solutions will
    # typically allow to fix more probabilities per outer loop of the leximin algorithm.
    model.setParam("Method", 2)  # optimize via barrier only
    model.setParam("Crossover", 0)  # deactivate cross-over

    return model, agent_vars, cap_var


def find_distribution_leximin(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              panel_size: int) \
        -> Tuple[List[FrozenSet[str]], List[float]]:

    grb.setParam("OutputFlag", 0)
    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: Set[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    new_committee_model, agent_vars, committees, covered_agents = _committee_generation(categories, people, panel_size)

    # Over the course of the algorithm, the selection probabilities of more and more agents get fixed to a certain value
    fixed_probabilities: Dict[str, float] = {}

    reduction_counter = 0

    # The outer loop maximizes the minimum of all unfixed probabilities while satisfying the fixed probabilities.
    # In each iteration, at least one more probability is fixed, but often more than one.
    while len(fixed_probabilities) < len(people):
        print(f"Fixed {len(fixed_probabilities)}/{len(people)} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(people, committees, fixed_probabilities)
        # In the inner loop, there is a column generation for maximizing the minimum of all unfixed probabilities
        while True:
            dual_model.optimize()
            if dual_model.status != GRB.OPTIMAL:
                for agent in fixed_probabilities:
                    # Relax all fixed probabilities by a small constant
                    fixed_probabilities[agent] = max(0., fixed_probabilities[agent] - 0.0001)
                    dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(people, committees,
                                                                                    fixed_probabilities)
                print(dual_model.status, f"REDUCE PROBS for {reduction_counter}th time.")
                reduction_counter += 1
                continue

            # Find the panel P for which Σ_{i ∈ P} yᵢ is largest, i.e., for which Σ_{i ∈ P} yᵢ ≤ ŷ is tightest
            agent_weights = {person: agent_var.x for person, agent_var in dual_agent_vars.items()}
            objective = sum(agent_weights[person] * agent_vars[person] for person in people)
            new_committee_model.setObjective(objective, GRB.MAXIMIZE)
            new_committee_model.optimize()
            new_set = _ilp_results_to_committee(agent_vars)  # panel P
            value = new_committee_model.objVal  # Σ_{i ∈ P} yᵢ

            upper = dual_cap_var.x  # ŷ
            dual_obj = dual_model.objVal  # ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ

            print(f"Maximin is at most {dual_obj - upper + value:.2%}, can do {dual_obj:.2%} with "
                                       f"{len(committees)} committees. Gap {value - upper:.2%}.")
            if value <= upper + 0.0005:
                # Within numeric tolerance, the panels in `committees` are enough to constrain the dual, i.e., they are
                # enough to support an optimal primal solution.
                for person, agent_weight in agent_weights.items():
                    if agent_weight > 0.0005 and person not in fixed_probabilities:
                        # `agent_weight` is the dual variable yᵢ of the constraint "Σ_{P : i ∈ P} x_P ≥ z" for
                        # i = `person` in the primal LP. If yᵢ is positive, this means that the constraint must be
                        # binding in all optimal solutions [1], and we can fix `person`'s probability to the
                        # optimal value of the primal/dual LP.
                        # [1] Theorem 3.3 in: Renato Pelessoni. Some remarks on the use of the strict complementarity in
                        # checking coherence and extending coherent probabilities. 1998.
                        fixed_probabilities[person] = max(0, dual_obj)
                break
            else:
                # Given that Σ_{i ∈ P} yᵢ > ŷ, the current solution to `dual_model` is not yet a solution to the dual.
                # Thus, add the constraint for panel P and recurse.
                assert new_set not in committees
                committees.add(new_set)
                dual_model.addConstr(grb.quicksum(dual_agent_vars[id] for id in new_set) <= dual_cap_var)

    # The previous algorithm computed the leximin selection probabilities of each agent and a set of panels such that
    # the selection probabilities can be obtained by randomizing over these panels. Here, such a randomization is found.
    primal = Model()
    # Variables for the output probabilities of the different panels
    committee_vars = [primal.addVar(vtype=GRB.CONTINUOUS, lb=0.) for _ in committees]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    eps = primal.addVar(vtype=GRB.CONTINUOUS, lb=0.)
    primal.addConstr(grb.quicksum(committee_vars) == 1)  # Probabilities add up to 1
    person_probs = []
    for person, prob in fixed_probabilities.items():
        person_probability = grb.quicksum(comm_var for committee, comm_var in zip(committees, committee_vars)
                                          if person in committee)
        person_probs.append(prob)
        primal.addConstr(person_probability >= prob - eps)
    primal.setObjective(eps, GRB.MINIMIZE)
    primal.optimize()

    # Bound variables between 0 and 1 and renormalize, because np.random.choice is sensitive to small deviations here
    probabilities = np.array([comm_var.x for comm_var in committee_vars]).clip(0, 1)
    probabilities = list(probabilities / sum(probabilities))

    return committees, probabilities, person_probs

"""
END of
Modification of https://github.com/pgoelz/citizensassemblies-replication/blob/master/stratification.py
"""

"""
ORDERED OUTCOMES METHOD FOR MAKING UNIFORMN LOTTERY
"""
def find_distribution_ordered_outcomes_uniform(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              panel_size: int, lottery_size: int, time_limit: int):
    N = len(people)
    EPS = 0.0005
    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.MIPGAP = EPS

    committees = [model.addMVar(N, vtype=GRB.BINARY) for i in range(lottery_size)] #list of committee vectors, if committee[i] is 1, then person 1 is in this committee

    # all committees must have correct number of members
    for committee in committees:
        model.addConstr(grb.quicksum(committee[i] for i in range(N)) == panel_size)
    
    # make sure the quotas are respected for all committees
    for feature in categories:
        for value in categories[feature]:
            new_vec = []
            for id, person in people.items():
                if person[feature] == value:
                    new_vec.append(1)
                else:
                    new_vec.append(0)
            for committee in committees:
                model.addConstr(grb.quicksum(new_vec[j]*committee[j] for j in range(N)) >= categories[feature][value]["min"]) # make sure lower quotas are respected
                model.addConstr(grb.quicksum(new_vec[j]*committee[j] for j in range(N)) <= categories[feature][value]["max"]) # make sure upper quotas are respected
    
    # ordered outcomes additional values
    t = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length N, one for each agent. All real numbers. 
    d = model.addMVar((N,N), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length N x N. Only positive values.

    # probability function for each person
    funcs = [grb.quicksum((committee[i]/lottery_size) for committee in committees) for i in range(N)] # the number of panels each person is part of
    model.addConstrs((t[k] + d[i][k] >= - funcs[i]) for i in range(N) for k in range(N)) # N**2 constraints

    isOptimized = True
    #start = time()
    # solve N single-objective optimization problems
    for i in range(N):
        objective = (i + 1) * t[i] + sum(d[j][i] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if (time() - start > time_limit):
            isOptimized = False
            break
        try:
            z = model.objVal
        except Exception as e:
            print("Exception occured when trying to retrieve objective value. ")
            isOptimized = False
            break
        model.addConstr(objective <= z + EPS)
    
    lottery = []

    for committee in committees:
        lottery.append(committee.x)
    
    return lottery, isOptimized # return the uniform lottery

"""
ORDERED OUTCOMES METHOD FOR PRODUCING A GENERAL LOTTERY OF PANELS

"""

# GENERAL LEXIMIN ORDERED OUTCOMES
# use N+1 panels, as this is the max number needed for an optimal solution
def find_distribution_ordered_outcomes_general(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              panel_size: int, time_limit: int):
    N = len(people)
    EPS = 0.0005

    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.MIPGAP = EPS*100
    
    committees = {i: model.addMVar(N, vtype=GRB.BINARY) for i in range(N+1)} # this is the maximum number of panels needed for an optimal solution
    committee_vars = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0.) for i in committees} #continous variables, probability value for each panel 
    
    # make sure panel size is correct
    for i in committees:
        model.addConstr(grb.quicksum(committees[i][j] for j in range(N)) == panel_size)
    
    # make sure total panel probability sums up to 1
    model.addConstr(grb.quicksum(committee_vars[i] for i in range(N+1)) == 1)

    # constraints for quota's
    for feature in categories:
        for value in categories[feature]:
            new_vec = []
            for id, person in people.items():
                if person[feature] == value:
                    new_vec.append(1)
                else:
                    new_vec.append(0)
            for i in committees:
                model.addConstr(grb.quicksum(new_vec[j]*committees[i][j] for j in range(N)) >= categories[feature][value]["min"])
                model.addConstr(grb.quicksum(new_vec[j]*committees[i][j] for j in range(N)) <= categories[feature][value]["max"])
    
    # ordered outcomes addition values as described by method
    t = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="t") # list of t variables of length N, one for each agent. All real numbers. 
    d = model.addMVar((N,N), vtype=GRB.CONTINUOUS, lb=0., name="d") # d matrix, of length N x N. Only positive values.
    
    # probability for each person, calculating its probability of being selected
    funcs = [grb.quicksum(committees[i][j]*committee_vars[i] for i in committees) for j in range(N)] # each persons probability value
    model.addConstrs((t[k] + d[i][k] >= - funcs[i]) for i in range(N) for k in range(N)) # N**2 constraints
    
    isOptimized = True
    start = time()
    # solve N single-objective optimization problems in lexicographic order
    for i in range(N):
        objective = (i + 1) * t[i] + sum(d[j][i] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if (time() - start > time_limit):
            isOptimized = False
            break
        try:
            z = model.objVal
        except Exception as e:
            isOptimized = False
            print("An exception occured when trying to retrieve the objective value. ")
            break
        model.addConstr(objective <= z + EPS)
    
    lottery = []

    for i in committees:
        lottery.append([committee_vars[i].x, committees[i].x])

    return lottery, isOptimized

def find_distribution_ordered_values_uniform(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              panel_size: int, lottery_size: int, time_limit: int):
    N = len(people)
    EPS = 0.0005
    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.MIPGAP = EPS

    committees = [model.addMVar(N, vtype=GRB.BINARY) for i in range(lottery_size)]
    
    for committee in committees:
        model.addConstr(grb.quicksum(committee[i] for i in range(N)) == panel_size)

    for feature in categories:
        for value in categories[feature]:
            new_vec = []
            for id, person in people.items():
                if person[feature] == value:
                    new_vec.append(1)
                else:
                    new_vec.append(0)
            for committee in committees:
                model.addConstr(grb.quicksum(new_vec[j]*committee[j] for j in range(N)) >= categories[feature][value]["min"])
                model.addConstr(grb.quicksum(new_vec[j]*committee[j] for j in range(N)) <= categories[feature][value]["max"])

    # additional variables for ordered values method. 
    h = model.addMVar((lottery_size,N), vtype=GRB.CONTINUOUS, lb=0., name="h") # h matrix

    # probability function for each person
    funcs = [grb.quicksum((committee[i]/lottery_size) for committee in committees) for i in range(N)]
    model.addConstrs((h[k, j] >= -(funcs[j] - (k/lottery_size))) for j in range(N) for k in range(lottery_size)) # N**2 constraints
    
    isOptimized = True
    start = time()
    # solve as many single-objective problems as there are possible function values - 1
    # (which in a uniform lottery is at most as many values as the size of the lottery)
    for k in range(lottery_size-1):
        print(k)
        objective = sum(h[k+1,j] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if (time() - start > time_limit):
            isOptimized = False
            break
        try:
            z = model.objVal
        except Exception as e:
            isOptimized = False
            print("An exception occured when trying to retrieve the objective value")
            break
        model.addConstr(objective <= z + EPS)
    
    lottery = []

    for committee in committees:
        lottery.append(committee.x)
    
    return lottery, isOptimized # return the uniform lottery

def find_distribution_ordered_values_valuelist(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              panel_size: int, lottery_size: int, value_list: List[float], time_limit: int):
    N = len(people)
    EPS = 0.0005
    grb.setParam("OutputFlag", 0)

    model = Model()
    model.Params.TimeLimit = time_limit
    # model.Params.MIPGAP = EPS

    committees = [model.addMVar(N, vtype=GRB.BINARY) for i in range(lottery_size)]
    
    for committee in committees:
        model.addConstr(grb.quicksum(committee[i] for i in range(N)) == panel_size)

    for feature in categories:
        for value in categories[feature]:
            new_vec = []
            for id, person in people.items():
                if person[feature] == value:
                    new_vec.append(1)
                else:
                    new_vec.append(0)
            for committee in committees:
                model.addConstr(grb.quicksum(new_vec[j]*committee[j] for j in range(N)) >= categories[feature][value]["min"])
                model.addConstr(grb.quicksum(new_vec[j]*committee[j] for j in range(N)) <= categories[feature][value]["max"])
    
    #ordered value additional variables
    h = model.addMVar((total_values,N), vtype=GRB.CONTINUOUS, lb=0., name="h") # h matrix
    
    funcs = [grb.quicksum((committee[i]/lottery_size) for committee in committees) for i in range(N)]

    model.addConstrs((h[k, j] >= - funcs[j] + value_list[k]) for j in range(N) for k in range(len(value_list))) # N**2 constraints
    
    isOptimized = True
    start = time()
    for k in range(len(value_list)-1):
        objective = sum(h[k+1,j] for j in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if (time() - start > time_limit):
            isOptimized = False
            break
        try:
            z = model.objVal
        except Exception as e:
            print("An exception occured when trying to retrieve the objective value. ")
            isOptimized = False
            break
        model.addConstr(objective <= z + EPS)
    
    lottery = []

    for committee in committees:
        lottery.append(committee.x)
    
    return lottery, isOptimized # return the uniform lottery

def calculate_person_probs_uniform(lottery):
    N = len(lottery) # number of committees
    M = len(lottery[0]) # number of people

    people_probs = [0]*M
    for i in range(N):
        for j in range(M):
            people_probs[j] += float(lottery[i][j]) / N
    return people_probs

def calculate_person_probs_general(lottery):
    N = len(lottery) # number of committees
    M = len(lottery[0][1]) # number of people

    people_probs = [0]*M

    for i in range(N):
        for j in range(M):
            people_probs[j] += (float(lottery[i][0]) * float(lottery[i][1][j]))
    return people_probs

if __name__ == '__main__':
    print("Running citizens' assembly example problems to compare original method to our new implementations")
    print("(and to show that our methods can solve these types of problems)")
    
    # Problem instance 1
    times = []
    people_probs = []
    categories, people = read_csv(f"data/citizensassembly/xsmall_categories.csv", f"data/citizensassembly/xsmall_data.csv")
    panelsize = 10
    lotterysize = 100
    timelimit = 3600

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "xsmall")

    # Problem instance 2
    times = []
    people_probs = []
    categories, people = read_csv(f"data/citizensassembly/small_categories.csv", f"data/citizensassembly/small_data.csv")
    panelsize = 20
    lotterysize = 100
    timelimit = 3600

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "small")

    # Problem instance 3
    times = []
    people_probs = []
    categories, people = read_csv(f"data/citizensassembly/medium_categories.csv", f"data/citizensassembly/medium_data.csv")
    panelsize = 37
    lotterysize = 100
    timelimit = 3600

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "medium_1")

    # Problem instance 4
    times = []
    people_probs = []
    categories, people = read_csv(f"data/citizensassembly/medium_2_categories.csv", f"data/citizensassembly/medium_2_data.csv")
    panelsize = 40
    lotterysize = 100
    timelimit = 3600

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "medium_2")


    # Problem instance 5
    times = []
    people_probs = []
    categories, people = read_csv("data/citizensassembly/large_categories.csv", "data/citizensassembly/large_data.csv")
    panelsize = 200
    lotterysize = 100
    timelimit = 7200

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "large_1")

     # Problem instance 5_2
    times = []
    people_probs = []
    categories, people = read_csv("data/citizensassembly/large_categories.csv", "data/citizensassembly/large_data.csv")
    panelsize = 200
    lotterysize = 10
    timelimit = 7200

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "large_11")

    # Problem instance 4
    times = []
    people_probs = []
    categories, people = read_csv("data/citizensassembly/large_2_categories.csv", "data/citizensassembly/large_2_data.csv")
    panelsize = 200
    lotterysize = 100
    timelimit = 7200

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "large_2")


    times = []
    people_probs = []
    categories, people = read_csv("data/citizensassembly/large_2_categories.csv", "data/citizensassembly/large_2_data.csv")
    panelsize = 200
    lotterysize = 50
    timelimit = 7200

    start = time()
    committees, probs, person_probs = find_distribution_leximin(categories, people, panelsize)
    times.append(time()-start)

    people_probs.append(person_probs)

    start = time()
    lottery, isOptimized = find_distribution_ordered_values_uniform(categories, people, panelsize, lotterysize, timelimit)
    times.append(time()-start)

    person_probs = calculate_person_probs_uniform(lottery)
    people_probs.append(person_probs)

    plot_fairness_statistics(people_probs[0], people_probs[1], times, "large_21")