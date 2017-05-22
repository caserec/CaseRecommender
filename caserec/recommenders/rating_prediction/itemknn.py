# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Item Based Collaborative Filtering Recommender

    Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that
    are similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.

    Literature:
        http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

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
        The number of item candidates strategy that you can choose for selecting the possible items to recommend.
    - space_type: string

"""

import numpy as np
from scipy.spatial.distance import squareform, pdist
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile
from caserec.recommenders.rating_prediction.base_knn import BaseKNNRecommenders
from caserec.utils.write_file import WriteFile

__author__ = 'Arthur Fortes'


class ItemKNN(BaseKNNRecommenders):
    def __init__(self, train_file, test_file, prediction_file=None, similarity_metric="correlation", neighbors=30,
                 space_type='\t'):
        self.train_set = ReadFile(train_file, space_type=space_type).return_information()
        self.test_set = ReadFile(test_file, space_type=space_type).return_information()
        BaseKNNRecommenders.__init__(self, self.train_set, self.test_set)
        self.k = neighbors
        self.prediction_file = prediction_file
        self.similarity_metric = similarity_metric
        self.predictions = list()
        self.si_matrix = None

    def compute_similarity(self):
        # Calculate distance matrix between items
        self.si_matrix = np.float32(squareform(pdist(self.matrix.T, self.similarity_metric)))
        # transform distances in similarities
        self.si_matrix = 1 - self.si_matrix
        del self.matrix

    '''
     for each pair (u,i) in test set, this method returns a prediction based
     on the others feedback in the train set

     rui = bui + (sum((ruj - buj) * sim(i,j)) / sum(sim(i,j)))

    '''
    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item_j in self.test['feedback'][user]:
                    list_n = list()
                    ruj = 0.0
                    try:
                        sum_sim = 0.0
                        for item_i in self.train['feedback'][user]:
                            try:
                                sim = self.si_matrix[self.map_items[item_i]][self.map_items[item_j]]
                                if np.math.isnan(sim):
                                    sim = 0.0
                            except KeyError:
                                sim = 0.0
                            list_n.append((item_i, sim))
                        list_n = sorted(list_n, key=lambda x: -x[1])

                        for pair in list_n[:self.k]:
                            try:
                                ruj += (self.train['feedback'][user][pair[0]] -
                                        self.bui[user][pair[0]]) * pair[1]
                                sum_sim += pair[1]
                            except KeyError:
                                pass

                        try:
                            ruj = self.bui[user][item_j] + (ruj / sum_sim)
                        except ZeroDivisionError:
                            ruj = self.bui[user][item_j]

                    except KeyError:
                        ruj = self.bui[user][item_j]

                    # normalize the ratings based on the highest and lowest value.
                    if ruj > self.train_set["max"]:
                        ruj = self.train_set["max"]
                    if ruj < self.train_set["min"]:
                        ruj = self.train_set["min"]
                    self.predictions.append((user, item_j, ruj))

            if self.prediction_file is not None:
                WriteFile(self.prediction_file, self.predictions).write_recommendation()
            return self.predictions

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > ItemKNN Algorithm]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])
        print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
              " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
        # training baselines bui
        self.fill_matrix()
        print("training time:: ", timed(self.train_baselines), " sec")
        self.compute_similarity()
        print("\nprediction_time:: ", timed(self.predict), " sec\n")
        self.evaluate(self.predictions)
