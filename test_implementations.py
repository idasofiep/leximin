import numpy as np
import typing
import random
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable, Optional, Set
from dataclasses import dataclass
from unittest import TestCase
from implementations import ordered_outcomes_allocation, ordered_values_allocation
from solver import create_integer_func_values, get_allocation_values
from analysis import makeRandomAllocation

@dataclass
class AllocationTestSet:
    instance: list[list[int]]
    expected_sorted_values: list[int]


allocation1 = AllocationTestSet(
    [[1,2,3],[2,2,9],[2,3,3],[2,2,2]],
    [0,2,3,9]
)

allocation2 = AllocationTestSet(
    [[2,1],[1,2]],
    [2,2]
)

allocation3 = AllocationTestSet(
    [[1,3],[1,2]],
    [1,3]
)

allocation4 = AllocationTestSet(
    [[4,3],[3,2]],
    [3,3]
)

allocation5 = AllocationTestSet(
    [[2,2,2,2],[1,1,1,1]],
    [2,4]
)

allocation6 = AllocationTestSet(
    [[3,3,3,3],[1,1,1,1]],
    [3,3]
)

allocation7 = AllocationTestSet(
    [[2,1,2,1],[1,2,1,2]],
    [4,4]
)

allocation8 = AllocationTestSet(
    [[3,4,2,1],[5,5,1,1]],
    [6,6]
)

allocation9 = AllocationTestSet(
    [[1,16,23,26,4,10,12,19,9],[1,16,22,26,4,9,13,20,9],[1,15,23,25,4,10,13,20,9]],
    [39,40,43]
)

allocation_sat_1 = AllocationTestSet(
    [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
    [1,1,1,1,1]
)

allocation_sat_2 = AllocationTestSet(
    [[4,3,2,1,0],[4,5,3,2,1],[2,3,6,5,3],[1,2,3,5,2],[2,3,4,5,6]],
    [4,5,5,6,6]
)

allocation_sat_3 = AllocationTestSet(
    [[10,9,8,7,6,5,4,3,2,1],[9,8,7,6,5,4,3,2,1,0]],
    [25,26]
)

class FindAllocationLeximinTests(TestCase):

    def test_allocation_oo(self):
        time_limit = 1000
        # test 1
        A, model = ordered_outcomes_allocation(allocation1.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation1.instance)
        self.assertEqual(sorted_values, allocation1.expected_sorted_values)

        # test 2
        A, model = ordered_outcomes_allocation(allocation2.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation2.instance)
        self.assertEqual(sorted_values, allocation2.expected_sorted_values)

        # test 3
        A, model = ordered_outcomes_allocation(allocation3.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation3.instance)
        self.assertEqual(sorted_values, allocation3.expected_sorted_values)

        # test 4
        A, model = ordered_outcomes_allocation(allocation4.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation4.instance)
        self.assertEqual(sorted_values, allocation4.expected_sorted_values)

        # test 5
        A, model = ordered_outcomes_allocation(allocation5.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation5.instance)
        self.assertEqual(sorted_values, allocation5.expected_sorted_values)

        # test 6
        A, model = ordered_outcomes_allocation(allocation6.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation6.instance)
        self.assertEqual(sorted_values, allocation6.expected_sorted_values)

        # test 7
        A, model = ordered_outcomes_allocation(allocation7.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation7.instance)
        self.assertEqual(sorted_values, allocation7.expected_sorted_values)

        # test 8
        A, model = ordered_outcomes_allocation(allocation8.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation8.instance)
        self.assertEqual(sorted_values, allocation8.expected_sorted_values)

        # test 9
        A, model = ordered_outcomes_allocation(allocation9.instance, time_limit)
        sorted_values = get_allocation_values(A, allocation9.instance)
        self.assertEqual(sorted_values, allocation9.expected_sorted_values)
    
    def test_allocation_ov(self):
        time_limit = 1000
        #test 1
        values_list = create_integer_func_values(allocation1.instance)
        A, model = ordered_values_allocation(allocation1.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation1.instance)
        self.assertEqual(sorted_values, allocation1.expected_sorted_values)

        #test 2
        values_list = create_integer_func_values(allocation2.instance)
        A, model = ordered_values_allocation(allocation2.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation2.instance)
        self.assertEqual(sorted_values, allocation2.expected_sorted_values)

        #test 3
        values_list = create_integer_func_values(allocation3.instance)
        A, model = ordered_values_allocation(allocation3.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation3.instance)
        self.assertEqual(sorted_values, allocation3.expected_sorted_values)

        #test 4
        values_list = create_integer_func_values(allocation4.instance)
        A, model = ordered_values_allocation(allocation4.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation4.instance)
        self.assertEqual(sorted_values, allocation4.expected_sorted_values)

        #test 5
        values_list = create_integer_func_values(allocation5.instance)
        A, model = ordered_values_allocation(allocation5.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation5.instance)
        self.assertEqual(sorted_values, allocation5.expected_sorted_values)

        #test 6
        values_list = create_integer_func_values(allocation6.instance)
        A, model = ordered_values_allocation(allocation6.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation6.instance)
        self.assertEqual(sorted_values, allocation6.expected_sorted_values)

        #test 7
        values_list = create_integer_func_values(allocation7.instance)
        A, model = ordered_values_allocation(allocation7.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation7.instance)
        self.assertEqual(sorted_values, allocation7.expected_sorted_values)

        #test 8
        values_list = create_integer_func_values(allocation8.instance)
        A, model = ordered_values_allocation(allocation8.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation8.instance)
        self.assertEqual(sorted_values, allocation8.expected_sorted_values)

        #test 9
        values_list = create_integer_func_values(allocation9.instance)
        A, model = ordered_values_allocation(allocation9.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation9.instance)
        self.assertEqual(sorted_values, allocation9.expected_sorted_values)

        # test sat_1
        values_list = create_integer_func_values(allocation_sat_1.instance)
        A, model = ordered_values_allocation(allocation_sat_1.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation_sat_1.instance)
        self.assertEqual(sorted_values, allocation_sat_1.expected_sorted_values)

        # test sat_2
        values_list = create_integer_func_values(allocation_sat_2.instance)
        A, model = ordered_values_allocation(allocation_sat_2.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation_sat_2.instance)
        self.assertEqual(sorted_values, allocation_sat_2.expected_sorted_values)

        # test sat_3
        values_list = create_integer_func_values(allocation_sat_3.instance)
        A, model = ordered_values_allocation(allocation_sat_3.instance, values_list, time_limit)
        sorted_values = get_allocation_values(A, allocation_sat_3.instance)
        self.assertEqual(sorted_values, allocation_sat_3.expected_sorted_values)
    
    def test_random_allocation(self):
        time_limit = 1000
        # test random allocation with 5 people and 5 items
        instance = makeRandomAllocation(5,5,1,10,1)
        A_oo, model_oo = ordered_outcomes_allocation(instance, time_limit)
        sorted_values_oo = get_allocation_values(A_oo, instance)

        values_list = create_integer_func_values(instance)
        A_ov, model_ov = ordered_values_allocation(instance, values_list, time_limit)
        sorted_values_ov = get_allocation_values(A_ov, instance)

        for i in range(len(sorted_values_oo)):
            self.assertAlmostEqual(sorted_values_oo[i], sorted_values_ov[i], places=2)
        
        # test random allocation with 5 people and 10 items
        instance = makeRandomAllocation(10,5,1,10,1)
        A_oo, model_oo = ordered_outcomes_allocation(instance, time_limit)
        sorted_values_oo = get_allocation_values(A_oo, instance)

        values_list = create_integer_func_values(instance)
        A_ov, model_ov = ordered_values_allocation(instance, values_list, time_limit)
        sorted_values_ov = get_allocation_values(A_ov, instance)

        for i in range(len(sorted_values_oo)):
            self.assertAlmostEqual(sorted_values_oo[i], sorted_values_ov[i], places=2)

