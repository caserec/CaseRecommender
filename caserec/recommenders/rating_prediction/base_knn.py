# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file is base for neighborhood-base algorithms

Parameters
-----------
    - train_set: dict
        train data dictionary generated for ReadFile(file).rating_prediction()
    - train_set: dict
        test data dictionary generated for ReadFile(file).rating_prediction()
"""

import numpy as np
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation

__author__ = 'Arthur Fortes'


class BaseKNNRecommenders(object):
    def __init__(self, train_set, test_set):
        self.train = train_set
        self.test = test_set
        self.regBi = 10
        self.regBu = 15
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()
        self.users = sorted(set(list(self.train['users']) + list(self.test['users'])))
        self.items = sorted(set(list(self.train['items']) + list(self.test['items'])))
        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.map_items = dict()
        self.map_users = dict()
        self.matrix = None

        for item_id, item in enumerate(self.items):
            self.map_items[item] = item_id

        for user_id, user in enumerate(self.users):
            self.map_users[user] = user_id

    def fill_matrix(self):
        self.matrix = np.zeros((self.number_users, self.number_items))
        for u, user in enumerate(self.users):
            try:
                for item in self.train['feedback'][user]:
                    self.matrix[u][self.map_items[item]] = self.train['feedback'][user][item]
            except KeyError:
                pass

    def train_baselines(self):
        for i in range(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def compute_bi(self):
        # bi = (rui - mi - bu) / (regBi + number of interactions)
        self.bi = dict()

        for item in self.train['items']:
            cont = 0
            # self.bi.update({item: 0})
            for user in self.train['di'][item]:
                self.bi[item] = self.bi.get(item, 0) + float(self.train['feedback'][user].get(item, 0)) - \
                                self.train['mean_rates'] - self.bu.get(user, 0)
                cont += 1
            if cont > 1:
                self.bi[item] = float(self.bi[item]) / float(self.regBi + cont)

    def compute_bu(self):
        # bu = (rui - mi - bi) / (regBu + number of interactions)
        self.bu = dict()
        for user in self.train['users']:
            cont = 0
            for item in self.train['du'][user]:
                self.bu[user] = self.bu.get(user, 0) + float(self.train['feedback'][user].get(item, 0)) - \
                                self.train['mean_rates'] - self.bi.get(item, 0)
                cont += 1
            if cont > 1:
                self.bu[user] = float(self.bu[user]) / float(self.regBu + cont)

    def compute_bui(self):
        # bui = mi + bu + bi
        for user in self.users:
            for item in self.items:
                try:
                    self.bui.setdefault(user, {}).update(
                        {item: self.train['mean_rates'] + self.bu[user] + self.bi[item]})
                except KeyError:
                    self.bui.setdefault(user, {}).update({item: self.train['mean_rates']})
        del self.bu
        del self.bi

    def evaluate(self, predictions):
        result = RatingPredictionEvaluation()
        res = result.evaluation(predictions, self.test)
        print("\nEval:: RMSE:", res[0], " MAE:", res[1])
