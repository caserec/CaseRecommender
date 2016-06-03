# coding=utf-8
from scipy.spatial.distance import squareform, pdist
from recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
import numpy as np

__author__ = 'Arthur Fortes'

'''

Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that are
similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.

More details: http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

'''


class ItemKNN(BaseRatingPrediction):
    def __init__(self, train_file, test_file='', space_type='\t', similarity_metric='correlation', k=30):
        BaseRatingPrediction.__init__(self, train_file, test_file, space_type=space_type)
        self.k = k
        self.similarity_metric = similarity_metric
        self.prediction_results = list()
        self.di_matrix = list()
        self.dict_nij = dict()
        self.fill_item_user_matrix()
        self.train_baselines()
        self.calculate_similarity()
        self.predict()

    def calculate_similarity(self):
        self.di_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))
        self.di_matrix = 1 - self.di_matrix

        for i, item_i in enumerate(self.dataset_items):
            for j, item_j in enumerate(self.dataset_items[i+1:]):
                b = list(self.train.item_interactions.get(item_i, {}))
                c = list(self.train.item_interactions.get(item_j, {}))
                nij = len(filter(set(b).__contains__, c))
                self.di_matrix[i][j+1] *= (float(nij)/float(nij + 100))
                self.di_matrix[j+1][i] = self.di_matrix[i][j+1]

        del self.matrix

    def predict_items(self, user, items_for_prediction):
        list_items = list()
        for item_i in items_for_prediction:
            sum_suv = 0
            sum_sim = 0
            total = list()
            i = self.dict_item_id[item_i]

            for item_j in self.train.user_interactions[user]:
                j = self.dict_item_id[item_j]
                if self.di_matrix[j][i] > 0:
                    total.append([item_j, self.di_matrix[j][i]])

            if len(total) > self.k:
                total.sort(key=lambda x: -x[1])

            for item_j in total[:self.k]:
                rate_j = float(self.train.user_interactions[user].get(item_j[0], 0))
                if rate_j != 0:
                    sum_suv += (rate_j - self.bui[user].get(item_j[0], 0)) * item_j[1]
                    sum_sim += item_j[1]
            try:
                try:
                    rui = self.bui[user][item_i] + (float(sum_suv) / float(sum_sim))
                except KeyError:
                    rui = 0

            except ZeroDivisionError:
                rui = self.bui[user][item_i]

            if rui != 0:
                if rui > 5.0:
                    rui = 5.0
                elif rui < 1:
                    rui = 1.0

                list_items.append([item_i, rui])
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
