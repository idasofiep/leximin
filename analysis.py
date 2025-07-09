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
from implementations import ordered_outcomes_allocation, ordered_values_allocation, ordered_outcomes_stratification_general, ordered_outcomes_stratification_uniform, ordered_values_stratification_uniform
from solver import create_integer_func_values, get_allocation_values, get_people_probabilities, get_people_probabilities_uniform
from math import floor
from math import ceil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import pandas as pd
import seaborn as sns


@dataclass
class AllocationAnalysis:
    people_number: int
    items_number: int
    min_value: int
    max_value: int
    solver_types: list[str]
    times: list[float]

@dataclass
class StratificationAnalysis:
    panel_size: int
    people_number: int
    number_of_panels: int
    quota_gap: int
    solver_types: list[str]
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
The method randomly assign each person to some of the categories,
so in theory most people will be part of about half of the categories.

The method return a people matrix, where each person is represented as a binary value for each category.
"""
def makeRandomStratification(people_number, cat_number):
    people = []
    for i in range(people_number):
        person = []
        for j in range(cat_number):
            value = random.choice([0,1])
            person.append(value)
        people.append(person)
    return people

    

"""
Method for plotting allocation statistics

- Takes a list of AllocationAnalysis objects as input

- makes a .csv file with all the data for each instance
- and .pdf files that show how the solving time change based on number of people and number of items.

"""
def plot_allocation_statistics(allocationAnalysisList: List[AllocationAnalysis], analysisName: str, time_limit: int):
    data = []
    plot_data = []
    N = len(allocationAnalysisList[0].solver_types) #number of solver types

    for analysis in allocationAnalysisList:
        data.append({"number of people": analysis.people_number, "number of items": analysis.items_number, "min value": analysis.min_value, "max value": analysis.max_value, "solver types: ": analysis.solver_types, "times": analysis.times})
        for i in range(N):
            plot_data.append({"number of people": analysis.people_number, "number of items": analysis.items_number, "min value": analysis.min_value, "max value": analysis.max_value, "solver type": analysis.solver_types[i], "time": min(analysis.times[i],time_limit), "time limit reached": (analysis.times[i] >= time_limit)})

    output_directory = Path(f"analysis_result/{analysisName}")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / "data.csv", index=False)

    df_plot = pd.DataFrame(plot_data)
    max_people = max(df["number of people"])
    min_people = min(df["number of people"])
    max_items = max(df["number of items"])
    min_items = min(df["number of items"])

    #PEOPLE PLOTS
    # line plot showing mean value and 95% confidence interval new color palette
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / "1_people_line.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / "2_people_line.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing one line for each number of items
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette="flare", col="solver type", hue="number of items", hue_norm=plc.LogNorm(), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "3_people_line.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing one line for each number of items, normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", col="solver type", hue="number of items", hue_norm=plc.LogNorm(), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "4_people_line.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing only ordered outcomes method, mean value and 95% confidence interval normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", col="solver type", hue="solver type", col_wrap = 1, kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "5_people_line.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot with color palette
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / "6_people_dot.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot with normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / "7_people_dot.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette="flare", col="solver type", hue="number of items", hue_norm=plc.LogNorm(), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "8_people_dot.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types, normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", col="solver type", hue="number of items", hue_norm=plc.LogNorm(), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "9_people_dot.pdf"
    people_fig.savefig(plot_path)
    plt.close()
    
    #ITEMS PLOTS
    # line plot showing mean value and 95 % confidence interval with normal colors
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    item_fig.ax.set_xlim(0.96*min_items, 1.10*max_items)
    item_fig.set_axis_labels("number of items", "time in seconds")
    plot_path = output_directory / "10_item_line.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval new color palette
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    item_fig.ax.set_xlim(0.96*min_items, 1.10*max_items)
    item_fig.set_axis_labels("number of items", "time in seconds")
    plot_path = output_directory / "11_item_line.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # line plot showing one line for each number of items
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", palette="flare", col="solver type", hue="number of people", hue_norm=plc.LogNorm(), kind="line", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "12_item_line.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # line plot showing one line for each number of items, normal colors
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", col="solver type", hue="number of people", hue_norm=plc.LogNorm(), kind="line", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "13_item_line.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95 % confidence interval with normal colors
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", col="solver type", hue="solver type", col_wrap=1, kind="line", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "14_item_line.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # dot plot with color palette
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    item_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    item_fig.set_axis_labels("number of items", "time in seconds")
    plot_path = output_directory / "15_item_dot.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # dot plot with normal colors
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", hue="solver type", hue_order=(allocationAnalysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    item_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    item_fig.set_axis_labels("number of items", "time in seconds")
    plot_path = output_directory / "16_people_dot.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", palette="flare", col="solver type", hue="number of people", hue_norm=plc.LogNorm(), style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "17_item_dot.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types
    item_fig = sns.relplot(data=df_plot, x="number of items", y="time", col="solver type", hue="number of people", hue_norm=plc.LogNorm(), style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=time_limit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / "18_item_dot.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    return


"""
Method for plotting allocation statistics

- Takes a list of AllocationAnalysis objects as input

- makes a .csv file with all the data for each instance
- and .pdf files that show how the solving time change based on number of people and number of items.

"""
def plot_stratification_statistics(analysisList: List[StratificationAnalysis], analysisName: str, timelimit: int):
    data = []
    plot_data = []

    N = len(analysisList[0].solver_types) #number of solver types

    for analysis in analysisList:
        data.append({"number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "panel size": analysis.panel_size, "solver types: ": analysis.solver_types, "times": analysis.times})
        for i in range(N):
            plot_data.append({"number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "panel size": analysis.panel_size,"solver type": analysis.solver_types[i], "time": min(analysis.times[i],timelimit), "time limit reached": (analysis.times[i] >= timelimit)})

    output_directory = Path(f"analysis_result")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{analysisName}_data.csv", index=False)

    df_plot = pd.DataFrame(plot_data)
    max_people = max(df["number of people"])
    min_people = min(df["number of people"])

    #PEOPLE PLOTS
    # line plot showing mean value and 95% confidence interval new color palette
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / f"{analysisName}_people_line_1.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / f"{analysisName}_people_line_2.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot with color palette
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / f"{analysisName}_people_dot_1.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types
    item_fig = sns.relplot(data=df_plot, x="number of people", y="time", col="solver type", style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / f"{analysisName}_people_dot_2.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    return

def plot_fairness_statistics(probabilityGeneral: List[float], probabilityUniform: List[float], times: List[float], analysisName: str):
    data = []
    plot_data = []
    difference_data = []

    probabilityGeneral.sort()
    probabilityUniform.sort()

    for i in range(len(probabilityGeneral)):
        difference = abs(probabilityGeneral[i] - probabilityUniform[i])
        data.append({"outcome position": i+1, "general probability": probabilityGeneral[i], "uniform probability": probabilityUniform[i], "difference": difference, "times": times})
        plot_data.append({"outcome position": i+1, "solution type": "general", "probability": probabilityGeneral[i]})
        plot_data.append({"outcome position": i+1, "solution type": "uniform", "probability": probabilityUniform[i]})

    output_directory = Path(f"analysis_result")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{analysisName}_fairness.csv", index=False)

    df_plot = pd.DataFrame(plot_data)
    max_people = len(probabilityGeneral) + 2
    min_people = 0

    #PEOPLE PLOTS
    # line plot showing mean value and 95% confidence interval new color palette
    prob_fig = sns.relplot(data=df_plot, x="outcome position", y="probability", hue="solution type", hue_order=(["general", "uniform"]), height=6, aspect=1.8)
    prob_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    prob_fig.ax.set_ylim(0, 1)
    prob_fig.set_axis_labels("outcome position", "people probabilities")
    plot_path = output_directory / f"{analysisName}_1_fairness.pdf"
    prob_fig.savefig(plot_path)
    plt.close()

    prob_fig = sns.relplot(data=df_plot, x="outcome position", y="probability", hue="solution type", hue_order=(["general", "uniform"]), kind="line", height=6, aspect=1.8)
    prob_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    prob_fig.ax.set_ylim(0, 1)
    prob_fig.set_axis_labels("outcome position", "people probabilities")
    plot_path = output_directory / f"{analysisName}_2_fairness.pdf"
    prob_fig.savefig(plot_path)
    plt.close()

    return

"""
Method for plotting allocation statistics

- Takes a list of AllocationAnalysis objects as input

- makes a .csv file with all the data for each instance
- and .pdf files that show how the solving time change based on number of people and number of items.

"""
def plot_uniform_stratification_statistics(analysisList: List[StratificationAnalysis], analysisName: str, timelimit: int):
    data = []
    plot_data = []

    N = len(analysisList[0].solver_types) #number of solver types

    for analysis in analysisList:
        data.append({"number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "panel size": analysis.panel_size, "solver types: ": analysis.solver_types, "times": analysis.times})
        for i in range(N):
            plot_data.append({"number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "panel size": analysis.panel_size,"solver type": analysis.solver_types[i], "time": min(analysis.times[i],timelimit), "time limit reached": (analysis.times[i] >= timelimit)})

    output_directory = Path(f"analysis_result")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{analysisName}_data.csv", index=False)

    df_plot = pd.DataFrame(plot_data)
    max_people = max(df["number of people"])
    min_people = min(df["number of people"])
    max_panels = max(df["number of panels"])
    min_panels = min(df["number of panels"])

    #PEOPLE PLOTS
    # line plot showing mean value and 95% confidence interval new color palette
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / f"{analysisName}_people_line_1.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval normal colors
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / f"{analysisName}_people_line_2.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot with color palette
    people_fig = sns.relplot(data=df_plot, x="number of people", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_people, 1.10*max_people)
    people_fig.set_axis_labels("number of people", "time in seconds")
    plot_path = output_directory / f"{analysisName}_people_dot_1.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types
    item_fig = sns.relplot(data=df_plot, x="number of people", y="time", col="solver type", style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / f"{analysisName}_people_dot_2.pdf"
    item_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval new color palette
    people_fig = sns.relplot(data=df_plot, x="number of panels", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_panels, 1.10*max_panels)
    people_fig.set_axis_labels("number of panels", "time in seconds")
    plot_path = output_directory / f"{analysisName}_panels_line_1.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval new color palette
    people_fig = sns.relplot(data=df_plot, x="number of panels", y="time", hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_panels, 1.10*max_panels)
    people_fig.set_axis_labels("number of panels", "time in seconds")
    plot_path = output_directory / f"{analysisName}_panels_line_2.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot with color palette
    people_fig = sns.relplot(data=df_plot, x="number of panels", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), style="time limit reached", height=6, aspect=1.8)
    people_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    people_fig.ax.set_xlim(0.96*min_panels, 1.10*max_panels)
    people_fig.set_axis_labels("number of panels", "time in seconds")
    plot_path = output_directory / f"{analysisName}_panels_dot_1.pdf"
    people_fig.savefig(plot_path)
    plt.close()

    # dot plot side by side solver types
    item_fig = sns.relplot(data=df_plot, x="number of panels", y="time", col="solver type", style="time limit reached", height=6, aspect=1.8)
    item_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    plot_path = output_directory / f"{analysisName}_panels_dot_2.pdf"
    item_fig.savefig(plot_path)
    plt.close()
    return

def plot_quota_stratification_statistics(analysisList: List[StratificationAnalysis], analysisName: str, timelimit: int):
    data = []
    plot_data = []

    N = len(analysisList[0].solver_types) #number of solver types

    for analysis in analysisList:
        data.append({"number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "panel size": analysis.panel_size, "quota gap": analysis.quota_gap, "solver types: ": analysis.solver_types, "times": analysis.times})
        for i in range(N):
            plot_data.append({"number of people": analysis.people_number, "number of panels": analysis.number_of_panels, "panel size": analysis.panel_size, "quota gap": analysis.quota_gap, "solver type": analysis.solver_types[i], "time": min(analysis.times[i],timelimit), "time limit reached": (analysis.times[i] >= timelimit)})

    output_directory = Path(f"analysis_result")

    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{analysisName}_data.csv", index=False)

    df_plot = pd.DataFrame(plot_data)
    max_quota = max(df["quota gap"])
    min_quota = min(df["quota gap"])

    #PEOPLE PLOTS
    # line plot showing mean value and 95% confidence interval new color palette
    quota_fig = sns.relplot(data=df_plot, x="quota gap", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    quota_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    quota_fig.ax.set_xlim(0.96*min_quota, 1.10*max_quota)
    quota_fig.set_axis_labels("quota gap", "time in seconds")
    plot_path = output_directory / f"{analysisName}_quota_line_1.pdf"
    quota_fig.savefig(plot_path)
    plt.close()

    # line plot showing mean value and 95% confidence interval new color palette
    quota_fig = sns.relplot(data=df_plot, x="quota gap", y="time", palette=sns.color_palette("husl", N), hue="solver type", hue_order=(analysisList[0].solver_types), kind="line", height=6, aspect=1.8)
    quota_fig.map(plt.axhline, y=timelimit, color=".7", dashes=(2, 1), zorder=0)
    quota_fig.ax.set_xlim(0.96*min_quota, 1.10*max_quota)
    quota_fig.set_axis_labels("quota gap", "time in seconds")
    plot_path = output_directory / f"{analysisName}_quota_line_1.pdf"
    quota_fig.savefig(plot_path)
    plt.close()

    return


if __name__ == '__main__':

    # ALLOCATION ANALYSIS
    # analyse ordered values and ordered outcomes method with different randomly generated discrete allocation problems
    print("Allocation analysis 1: ")

    oo_analysis_list = []
    ov_analysis_list = []
    all_analysis_list = []


    analysis_list = [] #store all allocation analysis objects
    time_limit = 120

    for i in range(10): # number of people 10
        for j in range(10): # number of items 10
            people = (i+1)*2
            items = (j+1)*2
            print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")

            instance = makeRandomAllocation(people,items,1,20,1)
            measured_times = []

            start = time()
            A, model = ordered_outcomes_allocation(instance, time_limit)
            stop = time()
            measured_times.append(stop-start)
            
            values_list = create_integer_func_values(instance)
            start = time()
            A, model = ordered_values_allocation(instance,values_list,time_limit)
            stop = time()
            measured_times.append(stop-start)

            analysis = AllocationAnalysis(people,items,1,20,["ordered outcomes", "ordered values"],measured_times)
            analysis_list.append(analysis)
            all_analysis_list.append(analysis)

    plot_allocation_statistics(analysis_list, "1", time_limit)
    
    # Analyse only ordered outcomes until timelimit
    print("Allocation analysis 2: ")
    analysis_list = [] #store all allocation analysis objects
    time_limit = 120

    for i in range(10):
        items = 10*(i+1)
        people = 5
        time_limit_reached = False
        while ((not time_limit_reached) and (people <= 100)):
            print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")

            instance = makeRandomAllocation(people,items,1,20,1)
            measured_times = []

            start = time()
            A, model = ordered_outcomes_allocation(instance, time_limit)
            stop = time()
            measured_times.append(stop-start)

            analysis = AllocationAnalysis(people,items,1,20,["ordered outcomes"],measured_times)
            analysis_list.append(analysis)
            oo_analysis_list.append(analysis)

            time_limit_reached = ((stop-start) >= time_limit) 
            people = people + 2

    plot_allocation_statistics(analysis_list, "2", time_limit)

    # Analyse only ordered values until timelimit
    print("Allocation analysis 3: ")
    analysis_list = [] #store all allocation analysis objects
    time_limit = 120

    for i in range(10):
        items = 10*(i+1)
        people = 5
        time_limit_reached = False
        while ((not time_limit_reached) and (people <= 100)):
            print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")

            instance = makeRandomAllocation(people,items,1,20,1)
            measured_times = []

            values_list = create_integer_func_values(instance)
            start = time()
            A, model = ordered_values_allocation(instance, values_list, time_limit)
            stop = time()
            measured_times.append(stop-start)

            analysis = AllocationAnalysis(people,items,1,20,["ordered values"],measured_times)
            analysis_list.append(analysis)
            ov_analysis_list.append(analysis)

            time_limit_reached = ((stop-start) >= time_limit) 
            people = people + 2

    plot_allocation_statistics(analysis_list, "3", time_limit)

    # Analyse only ordered outcomes until timelimit
    print("Allocation analysis 4: ")
    analysis_list = [] #store all allocation analysis objects
    time_limit = 120

    for i in range(10):
        people = 10*(i+1)
        items = 5
        time_limit_reached = False
        while ((not time_limit_reached) and (items <= 100)):
            print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")

            instance = makeRandomAllocation(people,items,1,20,1)
            measured_times = []

            start = time()
            A, model = ordered_outcomes_allocation(instance, time_limit)
            stop = time()
            measured_times.append(stop-start)

            analysis = AllocationAnalysis(people,items,1,20,["ordered outcomes"],measured_times)
            analysis_list.append(analysis)
            oo_analysis_list.append(analysis)

            time_limit_reached = ((stop-start) >= time_limit) 
            items = items + 2

    plot_allocation_statistics(analysis_list, "4", time_limit)

    # Analyse only ordered values until timelimit
    print("Allocation analysis 5: ")
    analysis_list = [] #store all allocation analysis objects
    time_limit = 120

    for i in range(10):
        people = 10*(i+1)
        items = 5
        time_limit_reached = False
        while ((not time_limit_reached) and (items <= 100)):
            print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")

            instance = makeRandomAllocation(people,items,1,20,1)
            measured_times = []

            values_list = create_integer_func_values(instance)
            start = time()
            A, model = ordered_values_allocation(instance, values_list, time_limit)
            stop = time()
            measured_times.append(stop-start)

            analysis = AllocationAnalysis(people,items,1,20,["ordered values"],measured_times)
            analysis_list.append(analysis)
            ov_analysis_list.append(analysis)

            time_limit_reached = ((stop-start) >= time_limit) 
            items = items + 2


    plot_allocation_statistics(analysis_list, "5", time_limit)

    # Analyse people = items until timeline for both methods
    print("Allocation analysis 6: ")
    analysis_list = [] #store all allocation analysis objects
    time_limit = 120

    people = 10
    items = 10
    time_limit_reached = False

    while ((not time_limit_reached) and (items <= 100)):
        print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")
        instance = makeRandomAllocation(people,items,1,20,1)
        measured_times = []

        values_list = create_integer_func_values(instance)
        start = time()
        A, model = ordered_values_allocation(instance, values_list, time_limit)
        stop = time()
        measured_times.append(stop-start)

        analysis = AllocationAnalysis(people,items,1,20,["ordered values"],measured_times)
        analysis_list.append(analysis)
        ov_analysis_list.append(analysis)

        time_limit_reached = ((stop-start) >= time_limit)
        people = people + 1
        items = items + 1

    plot_allocation_statistics(analysis_list, "6", time_limit)
    
    # Analyse people = items until timeline for both methods
    print("Allocation analysis 7: ")
    analysis_list = [] #store all allocation analysis objects
    time_limit = 120
    
    people = 10
    items = 10
    time_limit_reached = False
    
    while ((not time_limit_reached) and (items <= 100)):
        print("Analysing allocation instance with " + str(people) + " people and " + str(items) + " items")
        instance = makeRandomAllocation(people,items,1,20,1)
        measured_times = []

        start = time()
        A, model = ordered_outcomes_allocation(instance, time_limit)
        stop = time()
        measured_times.append(stop-start)

        analysis = AllocationAnalysis(people,items,1,20,["ordered outcomes"],measured_times)
        analysis_list.append(analysis)
        oo_analysis_list.append(analysis)

        time_limit_reached = ((stop-start) >= time_limit)
        people = people + 1
        items = items + 1


    plot_allocation_statistics(analysis_list, "7", time_limit)

    plot_allocation_statistics(oo_analysis_list, "8", time_limit)

    plot_allocation_statistics(ov_analysis_list, "9", time_limit)


    print("STRATIFICATION ANALYSIS WITH 2 CATS")

    #Analysis 1
    uniform_analysis_list = []
    analysis_list = []
    timelimit = 360
    lotterysize = 100 # for uniform lotteries

    for i in range(10):
        times = []

        len_people = 50 + i*5
        panelsize = floor(len_people/10) # panel is always 10% of the total number of people

        # we set quotas so that the panel should have around half of each category +- 2.
        upper = floor(panelsize/2) + 1
        lower = floor(panelsize/2) - 1
        quotas = [[lower, upper]]*2

        # make a random problem instance, which is the people pool of volunteers
        people = makeRandomStratification(len_people, 2)

        print("Running ordered outcomes method for a general lottery: ")
        start = time()
        panel_vars, panels, model = ordered_outcomes_stratification_general(people, panelsize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print("Running ordered outcomes method for uniform lottery")
        start = time()
        panels, model = ordered_outcomes_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print("Running ordered values method for uniform lottery")
        start = time()
        panels, model = ordered_values_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print(times)

        analysis = StratificationAnalysis(panelsize,len_people,lotterysize,2, ["oo_general", "oo_uniform", "ov_uniform"], times)
        analysis_list.append(analysis)
        uniform_analysis = StratificationAnalysis(panelsize,len_people,lotterysize,2, ["oo_uniform", "ov_uniform"], [times[1],times[2]])
        uniform_analysis_list.append(uniform_analysis)
    
    plot_stratification_statistics(analysis_list, "strat1", timelimit)

    # Analysis 1 continuing for uniform methods
    analysis_list = []
    lotterysize = 100 # for uniform lotteries

    for i in range(20):
        times = []

        len_people = 100 + i*5
        panelsize = floor(len_people/10) # panel is always 10% of the total number of people

        # we set quotas so that the panel should have around half of each category +- 2.
        upper = floor(panelsize/2) + 1
        lower = floor(panelsize/2) - 1
        quotas = [[lower, upper]]*2

        # make a random problem instance, which is the people pool of volunteers
        people = makeRandomStratification(len_people, 2)

        print("Running ordered outcomes method for uniform lottery")
        start = time()
        panels, model = ordered_outcomes_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print("Running ordered values method for uniform lottery")
        start = time()
        panels, model = ordered_values_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print(times)

        analysis = StratificationAnalysis(panelsize,len_people,lotterysize,2, ["oo_uniform", "ov_uniform"], times)
        uniform_analysis_list.append(analysis)
    
    plot_stratification_statistics(uniform_analysis_list, "uniform_strat1", timelimit)

    print("STRATIFICATION ANALYSIS WITH 2 CATS")

    analysis_list = []

    for i in range(10):

        lotterysize = 100 + i*25
        times = []

        len_people = 200
        panelsize = 20 # panel is always 10% of the total number of people

        # we set quotas so that the panel should have around half of each category +- 2.
        quotas = [[8, 12]]*2

        # make a random problem instance, which is the people pool of volunteers
        people = makeRandomStratification(len_people, 2)

        print("Running ordered outcomes method for uniform lottery")
        start = time()
        panels, model = ordered_outcomes_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print("Running ordered values method for uniform lottery")
        start = time()
        panels, model = ordered_values_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print(times)

        analysis = StratificationAnalysis(panelsize,len_people,lotterysize,4, ["oo_uniform", "ov_uniform"], times)
        analysis_list.append(analysis)
    
    plot_uniform_stratification_statistics(analysis_list, "strat2", timelimit)

    # Analysis of changing quotas for uniform methods

    analysis_list = []
    
    for i in range(10):

        lotterysize = 100
        times = []

        len_people = 200
        panelsize = 20 # panel is always 10% of the total number of people

        # we set quotas so that the panel should have around half of each category +- 2.
        quotas = [[10-i, 10+i]]*2 # quotas from 10-10 to 1-19

        quota_gap = i*2

        # make a random problem instance, which is the people pool of volunteers
        people = makeRandomStratification(len_people, 2)

        print("Running ordered outcomes method for uniform lottery")
        start = time()
        panels, model = ordered_outcomes_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print("Running ordered values method for uniform lottery")
        start = time()
        panels, model = ordered_values_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)
        stop = time()
        times.append(stop-start)

        print(times)

        analysis = StratificationAnalysis(panelsize,len_people,lotterysize,quota_gap,["oo_uniform", "ov_uniform"], times)
        analysis_list.append(analysis)
    
    plot_quota_stratification_statistics(analysis_list, "strat3", timelimit)


    # PLOT fairness analysis
    analysis_list = []
    timelimit = 200
    
    for i in range(1):
        lotterysize = 80
        times = []

        len_people = 80
        panelsize = 9 # panel is always 10% of the total number of people

        # we set quotas so that the panel should have around half of each category +- 2.
        quotas = [[3, 5]]*2 # quotas from 10-10 to 1-19

        # make a random problem instance, which is the people pool of volunteers
        people = makeRandomStratification(len_people, 2)

        print("Running ordered outcomes method for general lottery")
        panel_vars, panels, model = ordered_outcomes_stratification_general(people, panelsize, quotas, timelimit)

        gen_probs = get_people_probabilities(panel_vars, panels)

        print("Running ordered values method for uniform lottery")
        panels, model = ordered_values_stratification_uniform(people, panelsize, lotterysize, quotas, timelimit)

        uniform_probs = get_people_probabilities_uniform(panels)
        
        plot_fairness_statistics(gen_probs, uniform_probs, str(i+1))




    



    