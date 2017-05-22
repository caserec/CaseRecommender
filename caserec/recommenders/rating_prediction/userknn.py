# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

User Based Collaborative Filtering Recommender

    User-kNN predicts a user’s rating according to how similar users rated the same item. The algorithm matches similar
    users based on the similarity of their ratings on items.

    Literature:
        http://files.grouplens.org/papers/algs.pdf

Parameters
-----------
    - train_file: string
    - test_file: string
    - prediction_file: string
        file to write final prediction
    - similarity_metric: string
        Pairwise metric to compute the similarity between the users.
        Reference about distances:
            - http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    - neighbors: int
        The number of user candidates strategy that you can choose for selecting the possible items to recommend.
    - space_type: string

"""

import numpy as np
from scipy.spatial.distance import squareform, pdist
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile
from caserec.recommenders.rating_prediction.base_knn import BaseKNNRecommenders
from caserec.utils.write_file import WriteFile

__author__ = 'Arthur Fortes'


class UserKNN(BaseKNNRecommenders):
    def __init__(self, train_file, test_file, prediction_file=None, similarity_metric="correlation", neighbors=30,
                 space_type='\t'):
        self.train_set = ReadFile(train_file, space_type=space_type).return_information()
        self.test_set = ReadFile(test_file, space_type=space_type).return_information()
        BaseKNNRecommenders.__init__(self, self.train_set, self.test_set)
        self.k = neighbors
        self.similarity_metric = similarity_metric
        self.prediction_file = prediction_file
        self.predictions = list()
        self.su_matrix = None

    def compute_similarity(self):
        # Calculate distance matrix between users
        self.su_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))
        # transform distances in similarities
        self.su_matrix = 1 - self.su_matrix
        del self.matrix

    """
    for each pair (u,i) in test set, this method returns a prediction based
    on the others feedback in the train set

    rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

    """

    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item in self.test['feedback'][user]:
                    list_n = list()
                    try:
                        ruj = 0.0
                        sum_sim = 0.0

                        for user_j in self.train['di'][item]:
                            sim = self.su_matrix[self.map_users[user]][self.map_users[user_j]]
                            if np.math.isnan(sim):
                                sim = 0.0
                            list_n.append((user_j, sim))
                        list_n = sorted(list_n, key=lambda x: -x[1])

                        for pair in list_n[:self.k]:
                            try:
                                ruj += (self.train_set['feedback'][pair[0]][item] -
                                        self.bui[pair[0]][item]) * pair[1]
                                sum_sim += pair[1]
                            except KeyError:
                                pass

                        try:
                            ruj = self.bui[user][item] + (ruj / sum_sim)
                        except ZeroDivisionError:
                            ruj = self.bui[user][item]

                    except KeyError:
                        ruj = self.bui[user][item]

                    # normalize the ratings based on the highest and lowest value.
                    if ruj > self.train_set["max"]:
                        ruj = self.train_set["max"]
                    if ruj < self.train_set["min"]:
                        ruj = self.train_set["min"]

                    self.predictions.append((user, item, ruj))

            if self.prediction_file is not None:
                WriteFile(self.prediction_file, self.predictions).write_recommendation()
            return self.predictions

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > User Algorithm]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])
        print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
              " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
        # training baselines bui
        self.fill_matrix()
        print("training time:: ", timed(self.train_baselines), " sec")
        self.compute_similarity()
        print("prediction_time:: ", timed(self.predict), " sec\n")
        self.evaluate(self.predictions)
