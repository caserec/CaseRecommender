# coding=utf-8
from recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from utils.read_file import ReadFile
from utils.write_file import WriteFile

__author__ = 'Arthur Fortes'

'''

User-Attribute-kNN predicts a user’s rating according to how similar users rated the same item.
The algorithm matches similar users based on the similarity of their attributes scores.

More details: http://files.grouplens.org/papers/algs.pdf

'''


class UserAttributeKNN(BaseRatingPrediction):
    def __init__(self, train_file, distance_matrix_file, output_file, test_file=None, space_type="\t",
                 similarity_metric="correlation", k=30):
        BaseRatingPrediction.__init__(self, train_file, test_file, space_type=space_type)
        self.output_file = output_file
        self.distance_matrix_file = distance_matrix_file
        self.k = k
        self.similarity_metric = similarity_metric
        self.predictions = list()

        self.du_matrix = ReadFile(self.distance_matrix_file).read_matrix()
        del self.matrix

        # methods
        self.train_baselines()
        self.predict()
        WriteFile(self.output_file, self.predictions, self.space_type).write_prediction_file()

    def predict(self):
        for user in self.test_users:
            for item in self.test_feedback[user]:
                list_n = list()
                try:
                    ruj = 0
                    sum_sim = 0

                    for user_j in self.train_di[item]:
                        sim = self.du_matrix[self.map_users[user]][self.map_users[user_j]]
                        list_n.append((user_j, sim))
                    list_n = sorted(list_n, key=lambda x: -x[1])

                    for pair in list_n[:self.k]:
                        ruj += (self.train_feedback[pair[0]][item] - self.bui[pair[0]][item]) * pair[1]
                        sum_sim += pair[1]

                    ruj = self.bui[user][item] + (ruj / sum_sim)
                    if ruj > 5:
                        ruj = 5.0
                    if ruj < 0.5:
                        ruj = 0.5
                    self.predictions.append((user, item, ruj))

                except KeyError:
                    pass
