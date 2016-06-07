# coding=utf-8
import time
from scipy.spatial.distance import squareform, pdist
from recommenders.rating_prediction.base_KNN_recommenders import BaseKNNRecommenders
import numpy as np

__author__ = 'Arthur Fortes'

'''

Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that are
similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.

More details: http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

'''


class ItemKNN(BaseKNNRecommenders):
    def __init__(self, train_set, test_set, similarity_metric="correlation", neighbors=30):
        BaseKNNRecommenders.__init__(self, train_set, test_set)
        self.k = neighbors
        self.similarity_metric = similarity_metric
        self.predictions = list()

        self.di_matrix = np.float32(squareform(pdist(self.matrix.T, self.similarity_metric)))
        self.di_matrix = 1 - self.di_matrix
        del self.matrix

        # methods
        starting_point = time.time()
        self.train_baselines()
        elapsed_time = time.time() - starting_point
        print("- Training time: " + str(elapsed_time) + " second(s)")
        starting_point = time.time()
        self.predict()
        elapsed_time = time.time() - starting_point
        print("- Prediction time: " + str(elapsed_time) + " second(s)")

    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item_j in self.test['feedback'][user]:
                    list_n = list()
                    try:
                        ruj = 0
                        sum_sim = 0
                        for item_i in self.train['feedback'][user]:
                            try:
                                sim = self.di_matrix[self.map_items[item_i]][self.map_items[item_j]]
                            except KeyError:
                                sim = 0
                            list_n.append((item_i, sim))
                        list_n = sorted(list_n, key=lambda x: -x[1])

                        for pair in list_n[:self.k]:
                            ruj += (self.train['feedback'][user][pair[0]] - self.bui[user][pair[0]]) * pair[1]
                            sum_sim += pair[1]
                        ruj = self.bui[user][item_j] + (ruj / sum_sim)
                        if ruj > 5:
                            ruj = 5.0
                        if ruj < 0.5:
                            ruj = 0.5
                        self.predictions.append((user, item_j, ruj))
                    except KeyError:
                        pass
