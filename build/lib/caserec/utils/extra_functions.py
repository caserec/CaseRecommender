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


class ComputeBui(object):
    """
    Compute baselines based on training information considering information about users and items

    """
    def __init__(self, training_set):
        """

        :param training_set: Dictionary returned by ReadFile with method read()
        :type training_set: dict
        """
        self.training_set = training_set
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()

    def train_baselines(self):
        for i in range(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def compute_bi(self):
        # bi = (rui - mi - bu) / (regBi + number of interactions)
        self.bi = dict()

        for item in self.training_set['items']:
            cont = 0
            for user in self.training_set['users_viewed_item'][item]:
                self.bi[item] = self.bi.get(item, 0) + float(self.training_set['feedback'][user][item]) - \
                                self.training_set['mean_value'] - self.bu.get(user, 0)
                cont += 1
            if cont > 1:
                self.bi[item] = float(self.bi[item]) / float(10 + cont)

    def compute_bu(self):
        # bu = (rui - mi - bi) / (regBu + number of interactions)
        self.bu = dict()
        for user in self.training_set['users']:
            cont = 0
            for item in self.training_set['items_seen_by_user'][user]:
                self.bu[user] = self.bu.get(user, 0) + float(self.training_set['feedback'][user][item]) - \
                                self.training_set['mean_value'] - self.bi.get(item, 0)
                cont += 1
            if cont > 1:
                self.bu[user] = float(self.bu[user]) / float(15 + cont)

    def compute_bui(self):
        # bui = mi + bu + bi
        for user in self.training_set['users']:
            for item in self.training_set['items']:
                try:
                    self.bui.setdefault(user, {}).update(
                        {item: self.training_set['mean_value'] + self.bu[user] + self.bi[item]})
                except KeyError:
                    self.bui.setdefault(user, {}).update({item: self.training_set['mean_value']})

    def execute(self):
        self.train_baselines()
        return self.bui
