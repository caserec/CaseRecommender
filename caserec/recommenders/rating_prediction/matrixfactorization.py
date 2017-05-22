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
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile
from caserec.utils.write_file import WriteFile

__author__ = "Arthur Fortes"


class MatrixFactorization(object):
    def __init__(self, train_file, test_file, prediction_file=None, steps=30, learn_rate=0.01, delta=0.015, factors=10,
                 init_mean=0.1, init_stdev=0.1, baseline=False):
        self.train_set = ReadFile(train_file).return_information()
        self.test_set = ReadFile(test_file).return_information()
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

        self.users = sorted(set(list(self.train['users']) + list(self.test['users'])))
        self.items = sorted(set(list(self.train['items']) + list(self.test['items'])))
        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.map_items = dict()
        self.map_items_index = dict()
        self.map_users = dict()
        self.map_users_index = dict()
        for i, item in enumerate(self.items):
            self.map_items.update({item: i})
            self.map_items_index.update({i: item})
        for u, user in enumerate(self.users):
            self.map_users.update({user: u})
            self.map_users_index.update({u: user})
        self.regBi = 10
        self.regBu = 15
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()
        self.p = None
        self.q = None

        # methods
        self._create_factors()

    def _create_factors(self):
        self.p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2

    def _train_baselines(self):
        for i in range(10):
            self._compute_bi()
            self._compute_bu()
        self._compute_bui()

    def _compute_bi(self):
        # bi = (rui - mi - bu) / (regBi + number of interactions)
        self.bi = dict()

        for item in self.items:
            cont = 0
            for user in self.train['di'].get(item, []):
                self.bi[item] = self.bi.get(item, 0) + float(self.train['feedback'][user][item]) - \
                                self.train['mean_rates'] - self.bu.get(user, 0)
                cont += 1
            if cont > 1:
                self.bi[item] = float(self.bi[item]) / float(self.regBi + cont)

    def _compute_bu(self):
        # bu = (rui - mi - bi) / (regBu + number of interactions)
        self.bu = dict()
        for user in self.users:
            cont = 0
            for item in self.train['du'].get(user, []):
                self.bu[user] = self.bu.get(user, 0) + float(self.train['feedback'][user][item]) - \
                                self.train['mean_rates'] - self.bi.get(item, 0)
                cont += 1
            if cont > 1:
                self.bu[user] = float(self.bu[user]) / float(self.regBu + cont)

    def _compute_bui(self):
        # bui = mi + bu + bi
        for user in self.users:
            for item in self.items:
                self.bui.setdefault(user, {}).update(
                    {item: self.train['mean_rates'] + self.bu.get(user, 0.0) + self.bi.get(item, 0.0)})
        del self.bu
        del self.bi

    def _predict(self, user, item, u, i, cond=True):
        if self.baseline:
            rui = self.bui[user][item] + np.dot(self.p[u], self.q[i])
        else:
            rui = self.train["mean_rates"] + np.dot(self.p[u], self.q[i])

        if cond:
            if rui > self.train["max"]:
                rui = self.train["max"]
            elif rui < self.train["min"]:
                rui = self.train["min"]
        return rui

    def train_model(self):
        for step in range(self.steps):
            error_final = 0.0
            for u, user in enumerate(self.train["users"]):
                for item in self.train["feedback"][user]:
                    i = self.map_items[item]
                    eui = self.train["feedback"][user][item] - self._predict(user, item, u, i, False)
                    error_final += (eui ** 2.0)

                    # Adjust the factors
                    u_f = self.p[u]
                    i_f = self.q[i]

                    # Compute factor updates
                    delta_u = eui * i_f - self.delta * u_f
                    delta_i = eui * u_f - self.delta * i_f

                    # apply updates
                    self.p[u] += self.learn_rate * delta_u
                    self.q[i] += self.learn_rate * delta_i

            rmse = np.sqrt(error_final / self.train["ni"])
            print("step::", step, "RMSE::", rmse)

    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item in self.test['feedback'][user]:
                    u, i = self.map_users[user], self.map_items[item]
                    try:
                        self.predictions.append((user, item, self._predict(user, item, u, i, True)))
                    except KeyError:
                        if self.baseline:
                            self.predictions.append((user, item, self.bui[user][item]))
                        else:
                            self.predictions.append((user, item, self.train["mean_rates"]))

            if self.prediction_file is not None:
                self.predictions = sorted(self.predictions, key=lambda x: x[0])
                WriteFile(self.prediction_file, self.predictions).write_recommendation()
            return self.predictions

    def evaluate(self, predictions):
        result = RatingPredictionEvaluation()
        res = result.evaluation(predictions, self.test)
        print("Eval:: RMSE:" + str(res[0]) + " MAE:" + str(res[1]))

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > Matrix Factorization]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])
        print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
              " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
        # training baselines bui
        if self.baseline:
            print("training time:: ", timed(self._train_baselines) + timed(self.train_model), " sec")
        else:
            print("training time:: ", timed(self.train_model), " sec")
        print("\nprediction_time:: ", timed(self.predict), " sec\n")
        self.evaluate(self.predictions)
