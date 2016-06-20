# coding=utf-8
from recommenders.rating_prediction.itemknn import ItemKNN
from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'

"""

Item Based Collaborative Filtering Recommender with Attributes

Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that are
similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.

More details: http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

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
        The number of item candidates strategy that you can choose for selecting the possible items to recommend.

"""


class ItemAttributeKNN(ItemKNN):
    def __init__(self, train_set, test_set, distance_matrix_file, neighbors=30):
        ItemKNN.__init__(self, train_set, test_set, neighbors=neighbors)
        self.distance_matrix_file = distance_matrix_file

    def read_matrix(self):
        self.si_matrix = ReadFile(self.distance_matrix_file).read_matrix()
