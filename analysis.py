"""
This program will solve different allocation and stratification instance problems and analyse how
the different leximin methods performs compared to each other.
"""

import gurobipy as grb
from gurobipy import Model, GRB
import mip
import csv
import numpy as np
from typing import NewType, List
from dataclasses import dataclass
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import random
from time import time
from implementations import ordered_outcomes_allocation, ordered_values_allocation, saturation_allocation, ordered_outcomes_stratification, ordered_values_stratification, saturation_stratification
from solver import create_integer_func_values, get_allocation_values, get_stratification_probabilities
from math import ceil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class AllocationAnalysis:
    people_number: int
    items_number: int
    min_value: int
    max_value: int
    times: list[int]

@dataclass
class StratificationAnalysis:
    panel_size: int
    people_number: int
    number_of_panels: int
    times: list[int]

"""
Method for making random allocation instances.
"""
def makeRandomAllocation(people_number,items_number,lowest_value,highest_value, step):
    instance = []
    for i in range(people_number):
        values = []
        for j in range(items_number):
            value = random.randrange(lowest_value, highest_value, step)
            values.append(value)
        instance.append(values)
    return instance


"""
Method for making random stratification instances.

With large enough instances there is a random but even distribution over panels among people
and most probability distributions will be uniform. The method can be seen as a random generator
of panels with only one category so all people should have the same change to being chosen in a lottery.
"""
def makeRandomStratification(panel_size, people, panel_number):
    panel = []
    for i in range(panel_size):
        panel.append(1)
    for i in range(people - panel_size):
        panel.append(0)
    
    panels = []
    for i in range(panel_number):
        random.shuffle(panel)
        new_panel = panel.copy()
        panels.append(new_panel)
    return panels

"""
Method for plotting allocation statistics

- Takes a list of AllocationAnalysis objects as input

- makes a .csv file with all the data for each instance
- and .pdf files that show how the solving time change based on number of people and number of items.

"""
def plot_allocation_statistics(allocationAnalysisList: List[AllocationAnalysis], analysisName: str):
    data = []
    plot_data = []

    for analysis in allocationAnalysisList:
        data.append({"number of people": analysis.people_number, "number of items": analysis.items_number, "min value": analysis.min_value, "max value": analysis.max_value, "times": analysis.times})
        plot_data.append({"number of people": analysis.people_number, "number of items": analysis.items_number, "min value": analysis.min_value, "max value": analysis.max_value, "algorithm": "ordered outcomes", "time": analysis.times[0]})
        plot_data.append({"number of people": analysis.people_number, "number of items": analysis.items_number, "min value": analysis.min_value, "max value": analysis.max_value, "algorithm": "ordered values", "time": analysis.times[1]})

    output_directory = Path("analysis_result")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{analysisName}_analysis_data.csv", index=False)

    df_plot = pd.DataFrame(plot_data)

    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_people = max(df["number of people"])
    min_people = min(df["number of people"])
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time")
    plot_path = output_directory / f"{analysisName}_people_analysis_plot.pdf"
    people_fig.savefig(plot_path)

    df_plot = pd.DataFrame(plot_data)

    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_items = max(df["number of items"])
    min_items = min(df["number of items"])

    item_fig.ax.set_xlim(0.96*min_items, 1.10*max_items)
    item_fig.set_axis_labels("number of items", "time")

    plot_path = output_directory / f"{analysisName}_items_analysis_plot.pdf"
    item_fig.savefig(plot_path)
    return

"""
Method for plotting stratification statistics

- Takes a list of StratificationAnalysis objects as input

- makes a .csv file with all the data for each instance
- and .pdf files that show how the solving time change based on panel size and number of people.

"""
def plot_stratification_statistics(analysisList: List[StratificationAnalysis], analysisName: str):
    data = []
    plot_data = []

    for analysis in analysisList:
        data.append({"panel size": analysis.panel_size, "number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "times": analysis.times})
        plot_data.append({"panel size": analysis.panel_size, "number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "algorithm": "ordered outcomes", "time": analysis.times[0]})
        plot_data.append({"panel size": analysis.panel_size, "number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "algorithm": "ordered values", "time": analysis.times[1]})
        plot_data.append({"panel size": analysis.panel_size, "number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "algorithm": "saturation", "time": analysis.times[2]})

    output_directory = Path("analysis_result")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{analysisName}_analysis_data.csv", index=False)

    df_plot = pd.DataFrame(plot_data)

    panelsize_fig = sns.relplot(data=df_plot, x="panel size", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values", "saturation"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_size = max(df["panel size"])
    min_size = min(df["panel size"])

    panelsize_fig.ax.set_xlim(0.98*min_size, 1.02*max_size)
    panelsize_fig.set_axis_labels("panel size", "time")

    plot_path = output_directory / f"{analysisName}_panelsize_analysis_plot.pdf"
    panelsize_fig.savefig(plot_path)

    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values", "saturation"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_size = max(df["number of people"])
    min_size = min(df["number of people"])

    people_fig.ax.set_xlim(0.98*min_size, 1.02*max_size)
    people_fig.set_axis_labels("panel size", "time")

    plot_path = output_directory / f"{analysisName}_people_analysis_plot.pdf"
    people_fig.savefig(plot_path)
    return

if __name__ == '__main__':

    # ALLOCATION ANALYSIS
    # analyse ordered values and ordered outcomes method with different randomly generated discrete allocation problems

    print("Analyse Allocation problems, ")
    print("from 5 to 15 people and 5 to 15 items.")
    print("")
    print("timelimit for each instance is set to 120 seconds")

    analysis_list = [] #store all allocation analysis objects

    for i in range(10): # number of people
        for j in range(10): # number of items
            people = i+5
            items = j+5
            print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")

            instance = makeRandomAllocation(people,items,1,20,1)
            measured_times = []

            start = time()
            A, model = ordered_outcomes_allocation(instance)
            stop = time()
            measured_times.append(stop-start)
            
            values_list = create_integer_func_values(instance)
            start = time()
            A, model = ordered_values_allocation(instance,values_list)
            stop = time()
            measured_times.append(stop-start)

            analysis = AllocationAnalysis(people,items,1,20,measured_times)
            analysis_list.append(analysis)

    plot_allocation_statistics(analysis_list, "5-15")

    # STRATIFICATION ANALYSIS
    # analyse ordered values, ordered outcomes and saturation method with different randomly generated stratification problems

    print("Analyse Stratification problems, ")
    print("50, 55, ... 70 people")
    print("10, 12, ... 18 people in each panel")
    print("")
    print("timelimit for each instance is set to 120 seconds")

    analysis_list = [] #store all allocation analysis objects

    for i in range(5): # number of people
        for j in range(5): # number of panels
            people = (i*5)+50
            panelsize = (j*2)+10
            print("Analysing stratification instance with " + str(people) + " people and panel size: " + str(panelsize))
            print("and with 100 randomly generated panels")

            instance = makeRandomStratification(panelsize,people,100)

            measured_times = []

            start = time()
            X, model = ordered_outcomes_stratification(instance)
            stop = time()
            measured_times.append(stop-start)
            
            start = time()
            X, model = ordered_values_stratification(instance)
            stop = time()
            measured_times.append(stop-start)

            start = time()
            X, model = saturation_stratification(instance)
            stop = time()
            measured_times.append(stop-start)

            print(measured_times)

            analysis = StratificationAnalysis(panelsize,people,100,measured_times)
            analysis_list.append(analysis)

    plot_stratification_statistics(analysis_list, "30-100")




    



    