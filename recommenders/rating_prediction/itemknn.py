import time
from scipy.spatial.distance import squareform, pdist
from recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
import numpy as np

__author__ = 'Arthur Fortes'


class ItemKNN(BaseRatingPrediction):
    def __init__(self, train_file, test_file='', space_type='\t', similarity_metric='correlation', k=30):
        BaseRatingPrediction.__init__(self, train_file, test_file, space_type=space_type)
        self.k = k
        self.similarity_metric = similarity_metric
        self.prediction_results = list()
        self.di_matrix = list()
        self.fill_item_user_matrix()
        self.train_baselines()
        self.calculate_similarity()
        self.predict()

        print('Fill matrix and trained baselines...')

    def calculate_similarity(self):
        self.di_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))

        for i, item in enumerate(self.train.list_items):
            for j, item_j in enumerate(self.train.list_items[i+1:]):
                sim = 1 - self.di_matrix[i][j]
                if sim != 0:
                    # Intersection between users
                    list_i = set(self.train.item_interactions[item])
                    list_j = set(self.train.item_interactions[item_j])
                    nij = len(list_i & list_j)
                    sij = (float(nij)/float(nij + 100)) * sim
                    self.di_matrix[i][j] = sij
                    self.di_matrix[j][i] = sij

    def predict_items(self, user, items_for_interactions):
        list_items = list()

        for item_i in items_for_interactions:
            sum_suv = 0
            sum_sim = 0
            list_best = list()
            i = self.dict_item_id[item_i]

            for item_j in self.train.user_interactions[user]:
                j = self.dict_item_id[item_j]
                sim = self.di_matrix[i][j]
                if sim > 0:
                    list_best.append([item_j, sim])

            if len(list_best) > self.k:
                list_best.sort(key=lambda x: -x[1])

            for item_j in list_best:
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
        print('prediction')
        if self.test != '':
            for user in self.test.list_users:
                self.predict_items(user, self.test.user_interactions[user])
        else:
            for user in self.train.list_users:
                non_seen_items = list(set(self.train.list_items) - set(self.train.user_interactions[user].keys()))
                self.predict_items(user, non_seen_items)
                del self.bui[user]
                del self.train.user_interactions[user]


starting_point = time.time()
test = ItemKNN('C:\\Users\\Arthur\\Dropbox\\JournalWebSemantic\\ml_2k\\folds\\0\\train.dat',
               'C:\\Users\\Arthur\\Dropbox\\JournalWebSemantic\\ml_2k\\folds\\0\\test.dat')
elapsed_time = time.time() - starting_point
print("Runtime: " + str(elapsed_time / 60) + " second(s)\n")
