# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Matrix Factorization Based Collaborative Filtering Recommender

    Literature:
        Matrix Factorization Techniques for Recommender Systems
        http://dl.acm.org/citation.cfm?id=1608614

Parameters
-----------
    - train_file: string
    - test_file: string
    - prediction_file: string
        file to write final prediction
    - steps: int
         Number of steps over the training data
    - learn_rate: float
        Learning rate (alpha)
    - delta: float
        Regularization value
    - factors: int
        Number of latent factors per user/item
    - init_mean: float
        Mean of the normal distribution used to initialize the latent factors
    - init_stdev: float
        Standard deviation of the normal distribution used to initialize the latent factors
    - baseline: bool
        if True: Use the training data to build baselines (SVD Algorithm); else: Use only the mean

"""

import numpy as np
from framework.evaluation.rating_prediction import RatingPredictionEvaluation
from framework.utils.extra_functions import timed
from framework.utils.read_file import ReadFile
from framework.utils.write_file import WriteFile

__author__ = "Arthur Fortes"


class MatrixFactorization(object):
    def __init__(self, train_file, test_file, prediction_file=None, steps=30, learn_rate=0.01, delta=0.015, factors=10,
                 init_mean=0.1, init_stdev=0.1, baseline=False):
        self.train_set = ReadFile(train_file).rating_prediction()
        self.test_set = ReadFile(test_file).rating_prediction()
        self.prediction_file = prediction_file
        self.train = self.train_set
        self.test = self.test_set
        self.steps = steps
        self.learn_rate = learn_rate
        self.delta = delta
        self.factors = factors
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.baseline = baseline
        self.predictions = list()

        self.number_users = len(self.train["users"])
        self.number_items = len(self.train["items"])
        self.map_items = dict()
        self.map_items_index = dict()
        self.map_users = dict()
        self.map_users_index = dict()
        for i, item in enumerate(self.train["items"]):
            self.map_items.update({item: i})
            self.map_items_index.update({i: item})
        for u, user in enumerate(self.train["users"]):
            self.map_users.update({user: u})
            self.map_users_index.update({u: user})
        self.regBi = 10
        self.regBu = 15
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()
        self.p = None
        self.q = None
        self.final_matrix = None

        # methods
        self._create_factors()

    def _create_factors(self):
        self.p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2

    def _train_baselines(self):
        for i in xrange(10):
            self._compute_bi()
            self._compute_bu()
        self._compute_bui()

    def _compute_bi(self):
        # bi = (rui - mi - bu) / (regBi + number of interactions)
        self.bi = dict()

        for item in self.train['items']:
            cont = 0
            for user in self.train['di'][item]:
                self.bi[item] = self.bi.get(item, 0) + float(self.train['feedback'][user][item]) - \
                                self.train['mean_rates'] - self.bu.get(user, 0)
                cont += 1
            if cont > 1:
                self.bi[item] = float(self.bi[item]) / float(self.regBi + cont)

    def _compute_bu(self):
        # bu = (rui - mi - bi) / (regBu + number of interactions)
        self.bu = dict()
        for user in self.train['users']:
            cont = 0
            for item in self.train['du'][user]:
                self.bu[user] = self.bu.get(user, 0) + float(self.train['feedback'][user][item]) - \
                                self.train['mean_rates'] - self.bi.get(item, 0)
                cont += 1
            if cont > 1:
                self.bu[user] = float(self.bu[user]) / float(self.regBu + cont)

    def _compute_bui(self):
        # bui = mi + bu + bi
        for user in self.train['users']:
            for item in self.train['items']:
                self.bui.setdefault(user, {}).update({item: self.train['mean_rates'] + self.bu[user] + self.bi[item]})
        del self.bu
        del self.bi

    def _predict(self, user, item, cond=True):
        if self.baseline:
            rui = self.bui[self.map_users_index[user]][self.map_items_index[item]] + np.dot(self.p[user], self.q[item])
        else:
            rui = self.train["mean_rates"] + np.dot(self.p[user], self.q[item])

        if cond:
            if rui > self.train["max"]:
                rui = self.train["max"]
            elif rui < self.train["min"]:
                rui = self.train["min"]
        return rui

    def train_mf(self):
        for step in xrange(self.steps):
            error_final = 0.0
            for u, user in enumerate(self.train["users"]):
                for item in self.train["feedback"][user]:
                    eui = self.train["feedback"][user][item] - self._predict(u, self.map_items[item], False)
                    error_final += (eui ** 2.0)

                    # Adjust the factors
                    u_f = self.p[u]
                    i_f = self.q[self.map_items[item]]

                    # Compute factor updates
                    delta_u = eui * i_f - self.delta * u_f
                    delta_i = eui * u_f - self.delta * i_f

                    # apply updates
                    self.p[u] += self.learn_rate * delta_u
                    self.q[self.map_items[item]] += self.learn_rate * delta_i

            # rmse = np.sqrt(error_final / self.train["ni"])
            # print rmse

        self.final_matrix = np.dot(self.p, self.q.T)

    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item in self.test['feedback'][user]:
                    try:
                        self.predictions.append((user, item, self._predict(self.map_users[user],
                                                                           self.map_items[item]), True))
                    except KeyError:
                        self.predictions.append((user, item, self.train["mean_rates"]))

            if self.prediction_file is not None:
                WriteFile(self.prediction_file, self.predictions).write_prediction_file()
            return self.predictions

    def evaluate(self, predictions):
        result = RatingPredictionEvaluation()
        res = result.evaluation(predictions, self.test)
        print("Eval:: RMSE:" + str(res[0]) + " MAE:" + str(res[1]))

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > Matrix Factorization]\n")
        print("training data:: " + str(len(self.train_set['users'])) + " users and " + str(len(
            self.train_set['items'])) + " items and " + str(self.train_set['ni']) + " interactions")
        print("test data:: " + str(len(self.test_set['users'])) + " users and " + str(len(self.test_set['items'])) +
              " items and " + str(self.test_set['ni']) + " interactions")
        # training baselines bui
        if self.baseline:
            print("training time:: " + str(timed(self._train_baselines) + timed(self.train_mf))) + " sec"
        else:
            print("training time:: " + str(timed(self.train_mf))) + " sec"
        print("prediction_time:: " + str(timed(self.predict))) + " sec\n"
        self.evaluate(self.predictions)
