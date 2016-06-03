# coding=utf-8
from recommenders.rating_prediction.base_KNN_recommenders import BaseKNNRecommenders
from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'

'''

User-Attribute-kNN predicts a user’s rating according to how similar users rated the same item.
The algorithm matches similar users based on the similarity of their attributes scores.

More details: http://files.grouplens.org/papers/algs.pdf

'''


class UserAttributeKNN(BaseKNNRecommenders):
    def __init__(self, train_set, test_set, distance_matrix_file, similarity_metric="correlation", neighbors=30):
        BaseKNNRecommenders.__init__(self, train_set, test_set)
        self.distance_matrix_file = distance_matrix_file
        self.k = neighbors
        self.similarity_metric = similarity_metric
        self.predictions = list()

        self.du_matrix = ReadFile(self.distance_matrix_file).read_matrix()
        del self.matrix

        # methods
        self.train_baselines()
        self.predict()

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
