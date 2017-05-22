# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file has some functions to build some structures by attributes or training data. Methods:
    - read_dataset -> Simple Dictionary:: user:{item: feedback}
    - read_attribute -> Dictionary whit list:: category:[list of items]
    - compute_write_similarity_matrix -> compute similarity_matrix and write the result after

Parameters
-----------
    - input_file: string
    - space_type: string
        Divide data space type. E.g:
            '\t'  = tabular [default]
            ','   = comma
            '.'   = dot

"""

import numpy as np
from scipy.spatial.distance import squareform, pdist

__author__ = 'Arthur Fortes'


class TreatData(object):
    def __init__(self, input_file, type_space="\t"):
        self.input_file = input_file
        self.type_space = type_space
        self.item_category_dict = dict()
        self.dataset_dict = dict()
        self.category_list = set()

    def read_dataset(self):
        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.type_space)
                    user, item, feedback = inline[0], inline[1], inline[2]
                    self.dataset_dict.setdefault(user, {}).update({item: feedback})
    '''
    - category_file: string
        file that has categories and them respective items
    '''
    def read_attribute(self, category_file):
        with open(category_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.type_space)
                    inline[1] = inline[1].replace('\n', '')
                    self.category_list.add(inline[1])
                    self.item_category_dict.setdefault(inline[0], []).append(inline[1])
    '''
     - distance: string
        Pairwise metric to compute the similarity between the users/ items.
     - write_file: string
        file to write similarity matrix
     - user_matrix: bool
        if True build a matrix user x user; else build a matrix item x item
    '''
    def compute_write_distance_matrix(self, distance, write_file, user_matrix=True):
        lu = set()
        li = set()

        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.type_space)
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.dataset_dict.setdefault(user, {}).update({item: feedback})
                    lu.add(user)
                    li.add(item)

        matrix = np.zeros((len(lu), len(li)))
        lu = sorted(list(lu))
        li = sorted(list(li))

        map_items = dict()
        map_users = dict()
        for i, item in enumerate(li):
            map_items.update({item: i})
        for u, user in enumerate(lu):
            map_users.update({user: u})

        for user in self.dataset_dict:
            for item in self.dataset_dict[user]:
                matrix[map_users[user]][map_items[item]] = self.dataset_dict[user][item]

        if not user_matrix:
            matrix = matrix.T

        matrix = np.float32(squareform(pdist(matrix, distance)))
        matrix -= 1
        with open(write_file, "w") as infile:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    infile.write(str(matrix[i][j]) + self.type_space)
                infile.write("\n")
