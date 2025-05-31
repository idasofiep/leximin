"""
Program for solving problem instances using the leximin method. 
A problem instance should be defined as a .csv file in the data folder.
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
from time import time
from pathlib import Path
from implementations import ordered_outcomes_allocation, ordered_outcomes_stratification, ordered_values_allocation, ordered_values_stratification, saturation_allocation, saturation_stratification

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
Method to create integer function value list

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
Method for finding each person's probability of being chosen
"""

def get_stratification_probabilities(X, instance):
    people_probabilities = [0]*len(instance[0])
    for i in range(len(instance)):
        for j in range(len(instance[0])):
            people_probabilities[j] += round(X.x[i] * instance[i][j], 2)
    people_probabilities.sort()
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
Method for saving a stratification result as a .txt file in solver_results.
"""
def save_stratification_results(filename, X, instance, model, solvertype):
    result = open(Path("solver_results", f"{filename}_{solvertype}_stratification_result.txt"), "w", encoding="utf-8")
    result.write("\t" + "Stratification problem instance name: " + str(filename) + "\n")
    result.write("\t" + "Number of panels: " + str(len(instance)) + "\n")
    result.write("\t" + "Number of people: " + str(len(instance[0])) + "\n")
    result.write("\t\n")
    result.write("\t" + "Instance: " + "\n")
    for panel in instance:
        result.write("\t" + str(panel) +"\n")
    
    result.write("\t\n")
    result.write("\t Solved using " + solvertype + " method." + "\n")
    result.write("\t\n")
    result.write("\t" + "Panel probabilities: " + "\n")
    counter = 1
    for probability in X.x:
        result.write("\t Panel " + str(counter) + " should have a probability of: " + str(round(probability, 2)) + "\n")
        counter += 1
    result.write("\t\n")

    people_probabilities = get_stratification_probabilities(X, instance)
    result.write("\t Ordered Leximin Probability Values for all people: \n")
    result.write("\t" + str(people_probabilities) + "\n")
    result.close()
    return

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
        instance = read_csv("data/allocations/" + filename + ".csv")
        print("Solving Allocation problem instance located in data/allocations/" + filename + ".csv")

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo"):
            print("using the ordered outcomes method. ")
            A, model = ordered_outcomes_allocation(instance)
        
        #ORDERED VALUES METHOD
        elif (solvertype == "ov"):
            print("using the ordered values method. ")
            values_list = create_integer_func_values(instance)
            A, model = ordered_values_allocation(instance, values_list)
        
        #SATURATION METHOD
        elif (solvertype == "sat"):
            print("using the saturation method. ")
            A, model = saturation_allocation(instance)
        
        print("")
        save_allocation_results(filename, A, instance, model, solvertype)
        print("Results are saved in the solver_result folder. ")
    
    #STRATIFICATIONS
    elif (problemtype == "stratifications"):
        instance = read_csv("data/stratifications/" + filename + ".csv")
        print("Solving Stratification problem instance located in data/stratifications/" + filename + ".csv")

        #ORDERED OUTCOMES METHOD
        if (solvertype == "oo"):
            print("using the ordered outcomes method. ")
            X, model = ordered_outcomes_stratification(instance)
        
        #ORDERED VALUES METHOD
        elif (solvertype == "ov"):
            print("using the ordered values method. ")
            X, model = ordered_values_stratification(instance)

        #SATURATION METHOD
        elif (solvertype == "sat"):
            print("using the saturation method. ")
            X, model = saturation_stratification(instance)
        
        save_stratification_results(filename, X, instance, model, solvertype)