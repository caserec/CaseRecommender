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
    - metadata_file: string
        Metadata file ; Format file:
        item \t metadata \t value\n
    - similarity_matrix_file: file
        Pairwise metric to compute the similarity between the users based on a set of attributes.
        Format file:
        Distances separated by \t, where the users should be ordering. E g.:
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
    - ranking_file: string
        file to write final ranking
    - neighbors: int
        The number of user candidates strategy that you can choose for selecting the possible items to recommend.
    - similarity_metric: string
        Pairwise metric to compute the similarity between the users.
        Reference about distances:
            - http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
"""

import sys
from framework.recommenders.item_recommendation.userknn import UserKNN
from framework.utils.extra_functions import timed
from framework.utils.read_file import ReadFile

__author__ = "Arthur Fortes"


class UserAttributeKNN(UserKNN):
    def __init__(self, train_file, test_file=None, metadata_file=None, similarity_matrix_file=None, ranking_file=None,
                 neighbors=30, rank_number=10, similarity_metric="correlation"):
        UserKNN.__init__(self, train_file, test_file=test_file, ranking_file=ranking_file, neighbors=neighbors,
                         rank_number=rank_number, similarity_metric=similarity_metric)
        if metadata_file is None and similarity_matrix_file is None:
            print("This algorithm needs a similarity matrix or a matadata file!")
            sys.exit(0)

        if metadata_file is not None:
            self.metadata = ReadFile(metadata_file).read_metadata(self.users)
            self.matrix = self.metadata['matrix']
        self.similarity_matrix_file = similarity_matrix_file

    def read_matrix(self):
        self.su_matrix = ReadFile(self.similarity_matrix_file).read_matrix()

    def execute(self):
        print("[Case Recommender: Item Recommendation > User Attribute KNN Algorithm]\n")
        print("training data:: " + str(len(self.train_set["map_user"])) + " users and " + str(len(
            self.train_set["map_item"])) + " items and " + str(self.train_set["number_interactions"]) + " interactions")
        if self.test_file is not None:
            test_set = ReadFile(self.test_file).return_matrix()
            print("test data:: " + str(len(test_set["map_user"])) + " users and " + str(len(test_set["map_item"])) +
                  " items and " + str(test_set["number_interactions"]) + " interactions")
            del test_set
        if self.similarity_matrix_file is not None:
            print("training time:: " + str(timed(self.read_matrix))) + " sec"
        else:
            print("training time:: " + str(timed(self.compute_similarity))) + " sec"
        print("prediction_time:: " + str(timed(self.predict))) + " sec\n"
        if self.test_file is not None:
            self.evaluate()
