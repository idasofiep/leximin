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
    items_number: int
    people_number: int
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
def makeRandomAllocation(items_number,people_number,lowest_value,highest_value, step):
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
Method for making random stratification instances

The method can be seen as a random generator of panels with two categories.
"""
# cat 1 quota, cat 2 quota, people in cat 1, people in cat 2, number of panels
def makeRandomStratificationTwoCategories(panel_cat_1_size, panel_cat_2_size, people_number_cat_1, people_number_cat_2, panel_number):
    panel_1 = []
    for i in range(panel_cat_1_size):
        panel_1.append(1)
    for i in range(people_number_cat_1 - panel_cat_1_size):
        panel_1.append(0)
    
    panel_2 = []
    for i in range(panel_cat_2_size):
        panel_2.append(1)
    for i in range(people_number_cat_2 - panel_cat_2_size):
        panel_2.append(0)
    
    panels = []
    for i in range(panel_number):
        random.shuffle(panel_1)
        random.shuffle(panel_2)
        new_panel = panel_1.copy() + panel_2.copy()
        panels.append(new_panel)
    
    return panels

def plot_allocation_people_statistics(allocationAnalysisList: List[AllocationAnalysis], analysisName: str):
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

    fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_people = max(df["number of people"])
    min_people = min(df["number of people"])

    fig.ax.set_xlim(0.98*min_people, 1.02*max_people)
    fig.set_axis_labels("number of people", "time")

    plot_path = output_directory / f"{analysisName}_analysis_plot.pdf"
    fig.savefig(plot_path)
    return

def plot_allocation_items_statistics(allocationAnalysisList: List[AllocationAnalysis], analysisName: str):
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

    fig = sns.relplot(data=df_plot, x="number of items", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_items = max(df["number of items"])
    min_items = min(df["number of items"])

    fig.ax.set_xlim(0.98*min_items, 1.02*max_items)
    fig.set_axis_labels("number of items", "time")

    plot_path = output_directory / f"{analysisName}_analysis_plot.pdf"
    fig.savefig(plot_path)
    return

def plot_stratification_panelsize_statistics(analysisList: List[StratificationAnalysis], analysisName: str):
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

    fig = sns.relplot(data=df_plot, x="panel size", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values", "saturation"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_size = max(df["panel size"])
    min_size = min(df["panel size"])

    fig.ax.set_xlim(0.98*min_size, 1.02*max_size)
    fig.set_axis_labels("panel size", "time")

    plot_path = output_directory / f"{analysisName}_analysis_plot.pdf"
    fig.savefig(plot_path)
    return

def plot_stratification_panelnumber_statistics(analysisList: List[StratificationAnalysis], analysisName: str):
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

    fig = sns.relplot(data=df_plot, x="number of panels", y="time", hue="algorithm", hue_order=("ordered outcomes", "ordered values", "saturation"), drawstyle="steps-post", kind="line", height=6, aspect=1.8)
    max_size = max(df["number of panels"])
    min_size = min(df["number of panels"])

    fig.ax.set_xlim(0.98*min_size, 1.02*max_size)
    fig.set_axis_labels("number of panels", "time")

    plot_path = output_directory / f"{analysisName}_analysis_plot.pdf"
    fig.savefig(plot_path)
    return

if __name__ == '__main__':

    # ALLOCATION ANALYSIS
    # analyse ordered values and ordered outcomes method with different randomly generated discrete allocation problems
    """
    analysis_list = []

    # Number of people Analysis for OO and OV, from 5 to 30 people, with 10 items and values from 1 to 20

    for i in range(26):
        print("analyse instance with number of people: ", i+5)

        M = i+5
        N = 10
        P = 20

        instance = makeRandomAllocation(N,M,1,P,1)
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

        analysis = AllocationAnalysis(N,M,1,P,measured_times)
        analysis_list.append(analysis)
    
    plot_allocation_people_statistics(analysis_list, "number_of_people")

    analysis_list = []
    
    # Number of items Analysis for OO and OV, from 5 to 25 items, with 10 people and values from 1 to 20
    for i in range(16):
        print("analyse instance with number of items: ", i+5)

        M = 10
        N = 5+i
        P = 20

        instance = makeRandomAllocation(N,M,1,P,1)
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

        analysis = AllocationAnalysis(N,M,1,P,measured_times)
        analysis_list.append(analysis)

    plot_allocation_items_statistics(analysis_list, "number_of_items")
    """

    # STRATIFICATION ANALYSIS
    # Analyse different sized randomly generated stratification problems
    """
    analysis_list = []

    for i in range(20):
        print("analyse stratification problem with panel size ", (i+10))
        panel_size = i+10
        people_number = 50
        number_of_panels = 100

        instance = makeRandomStratification(panel_size, people_number, number_of_panels)

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

        analysis = StratificationAnalysis(panel_size,people_number,number_of_panels,measured_times)
        analysis_list.append(analysis)
    
    plot_stratification_panelsize_statistics(analysis_list, "stratification")
    """

    analysis_list = []
    # analyse two category instance
    # (panel_cat_1_size, panel_cat_2_size, people_number_cat_1, people_number_cat_2, panel_number)
    for i in range(10):
        print("analyse stratification problem with number of panels: ", ((i*2)+80))
        panel_cat_1_size = 10
        panel_cat_2_size = 20
        panel_size = 30
        people_number_cat_1 = 30
        people_number_cat_2 = 30
        people_number = 60
        number_of_panels = 80 + (i*2)

        instance = makeRandomStratificationTwoCategories(panel_cat_1_size, panel_cat_2_size, people_number_cat_1, people_number_cat_2, number_of_panels)

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

        analysis = StratificationAnalysis(panel_size,people_number,number_of_panels,measured_times)
        analysis_list.append(analysis)

    
    plot_stratification_panelnumber_statistics(analysis_list, "stratification_two_cat")




    