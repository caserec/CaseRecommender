# coding=utf-8
import time
from scipy.spatial.distance import squareform, pdist
from recommenders.rating_prediction.base_KNN_recommenders import BaseKNNRecommenders
import numpy as np

__author__ = 'Arthur Fortes'

'''

User-kNN predicts a user’s rating according to how similar users rated the same item. The algorithm matches similar
users based on the similarity of their ratings on items.

More details: http://files.grouplens.org/papers/algs.pdf

'''


class UserKNN(BaseKNNRecommenders):
    def __init__(self, train_set, test_set, similarity_metric="correlation", neighbors=30):
        print("\n[UserKNN] Number of Neighbors: " + str(neighbors) + " | "
                                                                     "Similarity Metric: " + str(similarity_metric))

        BaseKNNRecommenders.__init__(self, train_set, test_set)
        self.k = neighbors
        self.similarity_metric = similarity_metric
        self.predictions = list()

        self.du_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))
        self.du_matrix = 1 - self.du_matrix
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
                for item in self.test['feedback'][user]:
                    list_n = list()
                    try:
                        ruj = 0
                        sum_sim = 0

                        for user_j in self.train['di'][item]:
                            sim = self.du_matrix[self.map_users[user]][self.map_users[user_j]]
                            list_n.append((user_j, sim))
                        list_n = sorted(list_n, key=lambda x: -x[1])

                        for pair in list_n[:self.k]:
                            ruj += (self.train['feedback'][pair[0]][item] - self.bui[pair[0]][item]) * pair[1]
                            sum_sim += pair[1]

                        ruj = self.bui[user][item] + (ruj / sum_sim)
                        if ruj > 5:
                            ruj = 5.0
                        if ruj < 0.5:
                            ruj = 0.5
                        self.predictions.append((user, item, ruj))

                    except KeyError:
                        pass
