# coding=utf-8
from scipy.spatial.distance import squareform, pdist
from recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
import numpy as np

__author__ = 'Arthur Fortes'

'''

User-kNN predicts a user’s rating according to how similar users rated the same item. The algorithm matches similar
users based on the similarity of their ratings on items.

More details: http://files.grouplens.org/papers/algs.pdf

'''


class UserKNN(BaseRatingPrediction):
    def __init__(self, train_file, test_file='', space_type='\t', similarity_metric='correlation', k=30):
        BaseRatingPrediction.__init__(self, train_file, test_file, space_type=space_type)
        self.k = k
        self.similarity_metric = similarity_metric
        self.prediction_results = list()
        self.du_matrix = list()
        self.fill_user_item_matrix()
        self.train_baselines()
        self.calculate_similarity()
        self.predict()

    def calculate_similarity(self):
        self.du_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))

        for u, user in enumerate(self.train.list_users):
            for v, user_v in enumerate(self.train.list_users[u+1:]):
                sim = 1 - self.du_matrix[u][v]
                if sim != 0 or sim < 0:
                    # Intersection between users
                    list_u = set(self.train.user_interactions[user])
                    list_v = set(self.train.user_interactions[user_v])
                    nuv = len(list_u & list_v)
                    suv = (float(nuv)/float(nuv + 100)) * sim
                    self.du_matrix[u][v] = suv
                    self.du_matrix[v][u] = suv

        del self.matrix

    def predict_items(self, user, user_interactions):
        list_items = list()
        u = self.dict_user_id[user]
        total = sorted(range(len(self.du_matrix[u])), key=lambda m: -self.du_matrix[u][m])

        for item in user_interactions:
            sum_suv = 0
            sum_sim = 0

            for neighbor in total[:self.k]:
                rate = float(self.train.user_interactions[self.dataset_users[neighbor]].get(item, 0))
                if rate != 0:
                    sim = self.du_matrix[u][neighbor]
                    sum_suv += (rate - self.bui[self.dataset_users[neighbor]].get(item, 0)) * sim
                    sum_sim += sim
            try:
                try:
                    rui = self.bui[user][item] + (float(sum_suv) / float(sum_sim))
                except KeyError:
                    rui = 0

            except ZeroDivisionError:
                rui = self.bui[user][item]

            if rui != 0:
                if rui > 5.0:
                    rui = 5.0
                elif rui < 1:
                    rui = 1.0

                list_items.append([item, rui])
        self.prediction_results.append([user, list_items])

    def predict(self):
        if self.test != '':
            for user in self.test.list_users:
                self.predict_items(user, self.test.user_interactions[user])
        else:
            for user in self.train.list_users:
                non_seen_items = list(set(self.train.list_items) - set(self.train.user_interactions[user].keys()))
                self.predict_items(user, non_seen_items)
                del self.bui[user]
                del self.train.user_interactions[user]
