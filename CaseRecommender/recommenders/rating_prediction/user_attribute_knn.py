# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

User Based Collaborative Filtering Recommender with Attributes

    User-Attribute-kNN predicts a user’s rating according to how similar users rated the same item. The algorithm
    matches similar users based on the similarity of their attributes scores. However, instead of traditional UserKNN,
    this approach uses a pre-computed similarity matrix.

    Literature:
        More details: http://files.grouplens.org/papers/algs.pdf

Parameters
-----------
    - train_file: string
    - test_file: string
    - prediction_file: string
        file to write final prediction
    - similarity_matrix_file: file
        Pairwise metric to compute the similarity between the users based on a set of attributes.
        Format file:
        Distances separated by \t, where the users should be ordering. E g.:
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
    - neighbors: int
        The number of user candidates strategy that you can choose for selecting the possible items to recommend.

"""

from CaseRecommender.utils.extra_functions import timed
from CaseRecommender.utils.read_file import ReadFile
from CaseRecommender.recommenders.rating_prediction.userknn import UserKNN

__author__ = 'Arthur Fortes'


class UserAttributeKNN(UserKNN):
    def __init__(self, train_file, test_file, similarity_matrix_file, prediction_file=None, neighbors=30):
        UserKNN.__init__(self, train_file, test_file, prediction_file=prediction_file, neighbors=neighbors)
        self.similarity_matrix_file = similarity_matrix_file

    def read_matrix(self):
        self.su_matrix = ReadFile(self.similarity_matrix_file).read_matrix()

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > User Attribute KNN Algorithm]\n")
        print("training data:: " + str(len(self.train_set['users'])) + " users and " + str(len(
            self.train_set['items'])) + " items and " + str(self.train_set['ni']) + " interactions")
        print("test data:: " + str(len(self.test_set['users'])) + " users and " + str(len(self.test_set['items'])) +
              " items and " + str(self.test_set['ni']) + " interactions")
        # training baselines bui
        print("training time:: " + str(timed(self.train_baselines))) + " sec"
        self.read_matrix()
        print("prediction_time:: " + str(timed(self.predict))) + " sec\n"
        self.evaluate(self.predictions)
