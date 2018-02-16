# coding=utf-8
"""
    This file has some auxiliary functions for Case Recommender. Method:
        - check_error_file: check if file exist
        - check_len_lists: check if the size of two list are equal
        - timed: measure the execution time of a function
        - print_header: print header in the algorithms

"""

# © 2018. Case Recommender (MIT License)

import sys
import time

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


def check_error_file(file_check):
    """
    Function to check if file exist

    :param file_check: File to check
    :type file_check: str

    """

    try:
        open(file_check)
    except TypeError:
        raise TypeError("File cannot be empty or file is invalid: " + str(file_check))


def check_len_lists(list1, list2):
    """
    Function to check if 2 have the same length

    :param list1: First list
    :type list1: list

    :param list2: Second list
    :type list2: list

    """

    if len(list1) != len(list2):
        print("Error: Number of files in train list and rank list must be equal!")
        sys.exit()


def timed(f):
    """
    Function to calculate the time of execution

    :param f: Function name without ()
    :type f: function name

    :return: Time of execution
    :rtype: float

    """
    start = time.time()
    f()
    elapsed = time.time() - start
    return elapsed


def print_header(header_info, test_info=None):
    """
    Function to print the header with information of the files

    :param header_info: Dictionary with information about dataset or train file
    :type header_info: dict

    :param test_info: Dictionary with information about test file
    :type test_info: dict

    """

    print("[Case Recommender: %s]\n" % header_info['title'])
    print("train data:: %d users and %d items (%d interactions) | sparsity:: %.2f%%" %
          (header_info['n_users'], header_info['n_items'], header_info['n_interactions'], header_info['sparsity']))

    if test_info is not None:
        print("test data:: %d users and %d items (%d interactions) | sparsity:: %.2f%%\n" %
              (test_info['n_users'], test_info['n_items'], test_info['n_interactions'], test_info['sparsity']))

