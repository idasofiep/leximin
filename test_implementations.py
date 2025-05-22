import gurobipy as grb
from gurobipy import Model, GRB
import mip
import numpy as np
import typing
import random
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable, Optional, Set
from dataclasses import dataclass
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from copy import deepcopy
from itertools import combinations
from unittest import TestCase
from implementations import ordered_outcomes_allocation, ordered_values_allocation, saturation_allocation, create_integer_func_values, get_allocation_values

@dataclass
class AllocationTestSet:
    instance: list[list[int]]
    expected_sorted_values: list[int]
    unique_solution: bool


allocation1 = AllocationTestSet(
    [[1,2,3],[2,2,9],[2,3,3],[2,2,2]],
    [0,2,3,9],
    True
)

allocation2 = AllocationTestSet(
    [[2,1],[1,2]],
    [2,2],
    True
)

allocation3 = AllocationTestSet(
    [[1,3],[1,2]],
    [1,3],
    True
)

allocation4 = AllocationTestSet(
    [[4,3],[3,2]],
    [3,3],
    True
)

allocation5 = AllocationTestSet(
    [[2,2,2,2],[1,1,1,1]],
    [2,4],
    False
)

allocation6 = AllocationTestSet(
    [[3,3,3,3],[1,1,1,1]],
    [3,3],
    False
)

allocation7 = AllocationTestSet(
    [[2,1,2,1],[1,2,1,2]],
    [4,4],
    False
)

allocation8 = AllocationTestSet(
    [[3,4,2,1],[5,5,1,1]],
    [6,6],
    True
)

allocation9 = AllocationTestSet(
    [[1,16,23,26,4,10,12,19,9],[1,16,22,26,4,9,13,20,9],[1,15,23,25,4,10,13,20,9]],
    [39,40,43],
    False
)

allocation10 = AllocationTestSet(
    [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
    [1,1,1,1,1],
    True
)

class FindAllocationLeximinTests(TestCase):

    def test_allocation_oo(self):
        # test 1
        A, model = ordered_outcomes_allocation(allocation1.instance)
        sorted_values = get_allocation_values(A, allocation1.instance)
        self.assertEqual(sorted_values, allocation1.expected_sorted_values)

        # test 2
        A, model = ordered_outcomes_allocation(allocation2.instance)
        sorted_values = get_allocation_values(A, allocation2.instance)
        self.assertEqual(sorted_values, allocation2.expected_sorted_values)

        # test 3
        A, model = ordered_outcomes_allocation(allocation3.instance)
        sorted_values = get_allocation_values(A, allocation3.instance)
        self.assertEqual(sorted_values, allocation3.expected_sorted_values)

        # test 4
        A, model = ordered_outcomes_allocation(allocation4.instance)
        sorted_values = get_allocation_values(A, allocation4.instance)
        self.assertEqual(sorted_values, allocation4.expected_sorted_values)

        # test 5
        A, model = ordered_outcomes_allocation(allocation5.instance)
        sorted_values = get_allocation_values(A, allocation5.instance)
        self.assertEqual(sorted_values, allocation5.expected_sorted_values)

        # test 6
        A, model = ordered_outcomes_allocation(allocation6.instance)
        sorted_values = get_allocation_values(A, allocation6.instance)
        self.assertEqual(sorted_values, allocation6.expected_sorted_values)

        # test 7
        A, model = ordered_outcomes_allocation(allocation7.instance)
        sorted_values = get_allocation_values(A, allocation7.instance)
        self.assertEqual(sorted_values, allocation7.expected_sorted_values)

        # test 8
        A, model = ordered_outcomes_allocation(allocation8.instance)
        sorted_values = get_allocation_values(A, allocation8.instance)
        self.assertEqual(sorted_values, allocation8.expected_sorted_values)

        # test 9
        A, model = ordered_outcomes_allocation(allocation9.instance)
        sorted_values = get_allocation_values(A, allocation9.instance)
        self.assertEqual(sorted_values, allocation9.expected_sorted_values)
    
    def test_allocation_ov(self):
        #test 1
        values_list = create_integer_func_values(allocation1.instance)
        A, model = ordered_values_allocation(allocation1.instance, values_list)
        sorted_values = get_allocation_values(A, allocation1.instance)
        self.assertEqual(sorted_values, allocation1.expected_sorted_values)

        #test 2
        values_list = create_integer_func_values(allocation2.instance)
        A, model = ordered_values_allocation(allocation2.instance, values_list)
        sorted_values = get_allocation_values(A, allocation2.instance)
        self.assertEqual(sorted_values, allocation2.expected_sorted_values)

        #test 3
        values_list = create_integer_func_values(allocation3.instance)
        A, model = ordered_values_allocation(allocation3.instance, values_list)
        sorted_values = get_allocation_values(A, allocation3.instance)
        self.assertEqual(sorted_values, allocation3.expected_sorted_values)

        #test 4
        values_list = create_integer_func_values(allocation4.instance)
        A, model = ordered_values_allocation(allocation4.instance, values_list)
        sorted_values = get_allocation_values(A, allocation4.instance)
        self.assertEqual(sorted_values, allocation4.expected_sorted_values)

        #test 5
        values_list = create_integer_func_values(allocation5.instance)
        A, model = ordered_values_allocation(allocation5.instance, values_list)
        sorted_values = get_allocation_values(A, allocation5.instance)
        self.assertEqual(sorted_values, allocation5.expected_sorted_values)

        #test 6
        values_list = create_integer_func_values(allocation6.instance)
        A, model = ordered_values_allocation(allocation6.instance, values_list)
        sorted_values = get_allocation_values(A, allocation6.instance)
        self.assertEqual(sorted_values, allocation6.expected_sorted_values)

        #test 7
        values_list = create_integer_func_values(allocation7.instance)
        A, model = ordered_values_allocation(allocation7.instance, values_list)
        sorted_values = get_allocation_values(A, allocation7.instance)
        self.assertEqual(sorted_values, allocation7.expected_sorted_values)

        #test 8
        values_list = create_integer_func_values(allocation8.instance)
        A, model = ordered_values_allocation(allocation8.instance, values_list)
        sorted_values = get_allocation_values(A, allocation8.instance)
        self.assertEqual(sorted_values, allocation8.expected_sorted_values)

        #test 9
        values_list = create_integer_func_values(allocation9.instance)
        A, model = ordered_values_allocation(allocation9.instance, values_list)
        sorted_values = get_allocation_values(A, allocation9.instance)
        self.assertEqual(sorted_values, allocation9.expected_sorted_values)
    
    """
    Saturation method only for instances with unique solution
    """
    def test_allocation_sat(self):
        # test 1
        A, model = saturation_allocation(allocation1.instance)
        sorted_values = get_allocation_values(A, allocation1.instance)
        self.assertEqual(sorted_values, allocation1.expected_sorted_values)

        # test 2
        A, model = saturation_allocation(allocation2.instance)
        sorted_values = get_allocation_values(A, allocation2.instance)
        self.assertEqual(sorted_values, allocation2.expected_sorted_values)

        # test 3
        A, model = saturation_allocation(allocation3.instance)
        sorted_values = get_allocation_values(A, allocation3.instance)
        self.assertEqual(sorted_values, allocation3.expected_sorted_values)

        # test 4
        A, model = saturation_allocation(allocation4.instance)
        sorted_values = get_allocation_values(A, allocation4.instance)
        self.assertEqual(sorted_values, allocation4.expected_sorted_values)

        # test 8
        A, model = saturation_allocation(allocation8.instance)
        sorted_values = get_allocation_values(A, allocation8.instance)
        self.assertEqual(sorted_values, allocation8.expected_sorted_values)

        # test 10
        A, model = saturation_allocation(allocation10.instance)
        sorted_values = get_allocation_values(A, allocation10.instance)
        self.assertEqual(sorted_values, allocation10.expected_sorted_values)

    