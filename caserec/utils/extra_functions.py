# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file has some auxiliary functions for Case Recommender. Method:
    - check_error_file: check if file exist
    - check_len_lists: check if the number of two list are equal
    - timed: measure the execution time of a function

"""

import sys
import time

__author__ = 'Arthur Fortes'


def check_error_file(file_check):
    try:
        open(file_check)
    except IOError:
        print("Error: File cannot be empty or file is invalid: " + str(file_check))
        sys.exit()


def check_len_lists(list1, list2):
    if len(list1) != len(list2):
        print("Error: Number of files in train list and rank list must be equal!")
        sys.exit()


def timed(f):
    start = time.time()
    f()
    elapsed = time.time() - start
    return elapsed
