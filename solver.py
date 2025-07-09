"""
Program for solving problem specified by file input instances using the leximin method. 
A problem instance should be defined as a .csv file in the data folder.

The allocation or panel lottery result will be solved and saved in a .txt file in the solver_results folder.
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
import random
import math
from time import time
from pathlib import Path
from implementations import ordered_outcomes_allocation, ordered_outcomes_stratification_uniform, ordered_outcomes_stratification_general, ordered_values_allocation, ordered_values_stratification_uniform

"""
Read a csv file that define the problem instance
 - Allocation problems are defined as one M*N integer matrix (M = people, N = items)
 - Stratification problems are defined using two matrices, 
   one binary matrix for the people pool, 
   and one integer matrix for the category quotas

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
Method to create integer function value list

- Takes the problem instance as input
- Return a list of integers between 0 and the max value for the problem instance

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
Method for calculating a feasible panel size based on the quotas
Assumes that there are enough people in each category to enforce the qoutas
TODO: add check for feasibility and handling for infeasible quotas
"""
def calculate_panel_size(quotas):
    lower_bound = 0
    upper_bound = 0
    for i in range(len(quotas)):
        lower_bound += quotas[i][0]
        upper_bound += quotas[i][1]
    
    panel_size = lower_bound + math.floor((upper_bound - lower_bound)/2)
    return panel_size

"""
Method to get leximin sorted allocation values list
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
Method for finding each person's probability of being chosen in a general panel lottery
"""

def get_people_probabilities(panel_vars, panels):
    people_probabilities = [0]*len(panels[0].x)
    for i in range(len(panels)):
        for j in range(len(panels[0].x)):
            people_probabilities[j] += round(panel_vars.x[i] * panels[i][j].x, 4)
    return people_probabilities

"""
Method for finding each person's probability of being chosen in a uniform panel lottery
"""

def get_people_probabilities_uniform(panels):
    M = len(panels) # number of panels
    N = len(panels[0].x) # number of people

    people_probabilities = [0]*N
    for i in range(M):
        for j in range(N):
            people_probabilities[j] += round(panels[i][j].x/M, 4)
    return people_probabilities

"""
Method for saving a allocation result as a .txt file in solver_results.
"""
def save_allocation_results(filename, A, instance, model, solvertype):
    result = open(Path("solver_results", f"{filename}_{solvertype}_allocation_result.txt"), "w", encoding="utf-8")
    result.write("\t" + "Allocation problem instance name: " + str(filename) + "\n")
    result.write("\t" + "Number of agents: " + str(len(instance)) + "\n")
    result.write("\t" + "Number of items: " + str(len(instance[0])) + "\n")
    result.write("\t\n")
    result.write("\t" + "Instance: " + "\n")
    for agent in instance:
        result.write("\t" + str(agent) +"\n")
    
    result.write("\t\n")
    result.write("\t Solved using " + solvertype + " method." + "\n")
    result.write("\t\n")
    result.write("\t" + "Result allocation: " + "\n")
    for agent in A.X:
        new_list = []
        for val in agent:
            new_list.append(int(val))
        result.write("\t" + str(new_list) + "\n")
    result.write("\t\n")

    values = get_allocation_values(A, instance)
    result.write("\t Ordered Leximin Values: \n")
    result.write("\t" + str(values) + "\n")
    result.close()
    return
"""
Method for saving a stratification result, a uniform lottery of panels, as a .txt file in solver_results.
"""
def save_stratification_results_uniform_lottery(filename, panels, model, solvertype):
    N = len(panels) # number of panels
    M = len(panels[0].x) #number of people
    result = open(Path("solver_results", f"{filename}_{solvertype}_stratification_result.txt"), "w", encoding="utf-8")
    result.write("\t" + "Stratification problem instance name: " + str(filename) + "\n")
    result.write("\t" + "Number of panels: " + str(N) + "\n")
    result.write("\t" + "Number of people: " + str(M) + "\n")
    result.write("\t\n")
    result.write("\t Solved using " + solvertype + " method." + "\n")
    result.write("\t\n")
    result.write("\t" + " Lottery Result [Panel] :" + "\n")

    for i in range(N):
        this_panel = []
        for j in range(M):
            this_panel.append(int(panels[i][j].x))
        result.write("\t " + str(this_panel) + " \n")
        result.write("\t\n")
    
    people_probabilities = get_people_probabilities_uniform(panels)

    result.write("\t People probabilities [personID, probability]: \n")
    for i in range(M):
        result.write("\t [" + str(i+1) + ", " + str(people_probabilities[i]) + "] \n")
    return

"""
Method for saving a stratification result as a .txt file in solver_results.
"""
def save_stratification_results_general_lottery(filename, panel_vars, panels, model, solvertype):
    N = len(panels) # number of panels
    M = len(panels[0].x) #number of people
    result = open(Path("solver_results", f"{filename}_{solvertype}_stratification_result.txt"), "w", encoding="utf-8")
    result.write("\t" + "Stratification problem instance name: " + str(filename) + "\n")
    result.write("\t" + "Number of panels: " + str(N) + "\n")
    result.write("\t" + "Number of people: " + str(M) + "\n")
    result.write("\t\n")
    result.write("\t Solved using " + solvertype + " method." + "\n")
    result.write("\t\n")
    result.write("\t" + " Lottery Result: [PanelID, Panel probabilitiy, Panel] " + "\n")

    for i in range(N):
        this_panel = []
        for j in range(M):
            this_panel.append(int(panels[i][j].x))
        result.write("\t [" + str(i+1) + ", " + str(panel_vars[i].x) + ", " + str(this_panel) + "] \n")
        result.write("\t\n")
    
    people_probabilities = get_people_probabilities(panel_vars, panels)

    result.write("\t People probabilities [personID, probability]: \n")
    for i in range(M):
        result.write("\t [" + str(i+1) + ", " + str(people_probabilities[i]) + "] \n")
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("problemtype", type=str, help="problem type, either allocation or stratification(panel lottery generation)")
    parser.add_argument("solvertype", type=str, help="solver algorithm, either oo (ordered outcomes) or ov (ordered values)")
    parser.add_argument("filename", type=str, help="filename input (str)")
    args = parser.parse_args()
    filename = args.filename
    problemtype = args.problemtype
    solvertype = args.solvertype
    timelimit = 1200 # 20 minutes
    
    #ALLOCATIONS
    if (problemtype == "allocations"):
        instance = read_csv("data/allocations/" + filename + ".csv")
        print("Solving Allocation problem instance located in data/allocations/" + filename + ".csv")
        print("time limit is set to 20 minutes. ")

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo"):
            print("using the ordered outcomes method. ")
            A, model = ordered_outcomes_allocation(instance, timelimit)
        
        #ORDERED VALUES METHOD
        elif (solvertype == "ov"):
            print("using the ordered values method. ")
            valueslist = create_integer_func_values(instance)
            A, model = ordered_values_allocation(instance, valueslist, timelimit)
        
        save_allocation_results(filename, A, instance, model, solvertype)
        print("Results are saved in the solver_result folder. ")
    
    #STRATIFICATIONS
    elif (problemtype == "stratifications"):
        people = read_csv("data/stratifications/" + filename + "_people.csv")
        quotas = read_csv("data/stratifications/" + filename + "_quotas.csv")

        print("Enter panel size: ")
        panelsize = int(input())

        print("Solving " + filename + " stratification problem instance located in data/stratifications/")
        print("Time limit is set to 20 minutes. ")

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo_uniform"):
            print("Enter lottery size for the uniform lottery: ")
            lotterysize = int(input())

            print("using the ordered outcomes method for uniform lottery. ")
            panels, model = ordered_outcomes_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
            save_stratification_results_uniform_lottery(filename, panels, model, solvertype)
        
        elif (solvertype == "oo_general"):
            print("using the ordered outcomes method for general lottery. ")
            panel_vars, panels, model = ordered_outcomes_stratification_general(people, panelsize, quotas, timelimit)
            save_stratification_results_general_lottery(filename, panel_vars, panels, model, solvertype)
        
        #ORDERED VALUES METHOD
        elif (solvertype == "ov_uniform"):
            print("Enter lottery size for the uniform lottery: ")
            lotterysize = int(input())
            print("using the ordered values method for uniform lottery. ")
            panels, model = ordered_values_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
            save_stratification_results_uniform_lottery(filename, panels, model, solvertype)
        




        






    
