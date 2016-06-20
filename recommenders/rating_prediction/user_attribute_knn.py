# coding=utf-8
from recommenders.rating_prediction.userknn import UserKNN
from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'

"""

User Based Collaborative Filtering Recommender with Attributes

User-Attribute-kNN predicts a user’s rating according to how similar users rated the same item.
The algorithm matches similar users based on the similarity of their attributes scores.

More details: http://files.grouplens.org/papers/algs.pdf

This algorithm accepts a precomputed distance matrix instead compute it inside of its code

Parameters
-----------
    distance_matrix_file: file
        Pairwise metric to compute the similarity between the users based on a set of attributes.
        Format file:
        Distances separated by \t, where the users should be ordering. E g.:
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n

    neighbors: int
        The number of user candidates strategy that you can choose for selecting the possible items to recommend.

"""


class UserAttributeKNN(UserKNN):
    def __init__(self, train_set, test_set, distance_matrix_file, neighbors=30):
        UserKNN.__init__(self, train_set, test_set, neighbors=neighbors)
        self.distance_matrix_file = distance_matrix_file

    def read_matrix(self):
        self.su_matrix = ReadFile(self.distance_matrix_file).read_matrix()
