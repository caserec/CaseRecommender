# coding=utf-8
from recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from utils.read_file import ReadFile
from utils.write_file import WriteFile

__author__ = 'Arthur Fortes'

'''

Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that are
similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.

More details: http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

'''


class ItemAttributeKNN(BaseRatingPrediction):
    def __init__(self, train_file, distance_matrix_file, output_file, test_file=None, space_type="\t",
                 similarity_metric="correlation", k=30):
        BaseRatingPrediction.__init__(self, train_file, test_file, space_type=space_type)
        self.output_file = output_file
        self.k = k
        self.distance_matrix_file = distance_matrix_file
        self.similarity_metric = similarity_metric
        self.predictions = list()

        self.di_matrix = ReadFile(self.distance_matrix_file).read_matrix()
        del self.matrix

        # methods
        self.train_baselines()
        self.predict()
        WriteFile(self.output_file, self.predictions, self.space_type).write_prediction_file()

    def predict(self):
        for user in self.test_users:
            for item_j in self.test_feedback[user]:
                list_n = list()
                try:
                    ruj = 0
                    sum_sim = 0
                    for item_i in self.train_feedback[user]:
                        try:
                            sim = self.di_matrix[self.map_items[item_i]][self.map_items[item_j]]
                        except KeyError:
                            sim = 0
                        list_n.append((item_i, sim))
                    list_n = sorted(list_n, key=lambda x: -x[1])

                    for pair in list_n[:self.k]:
                        ruj += (self.train_feedback[user][pair[0]] - self.bui[user][pair[0]]) * pair[1]
                        sum_sim += pair[1]
                    ruj = self.bui[user][item_j] + (ruj / sum_sim)
                    if ruj > 5:
                        ruj = 5.0
                    if ruj < 0.5:
                        ruj = 0.5
                    self.predictions.append((user, item_j, ruj))
                except KeyError:
                    pass
