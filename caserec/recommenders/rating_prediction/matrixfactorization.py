# coding=utf-8
"""
© 2017. Case Recommender All Rights Reserved (License GPL3)

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
    - bias_learn_rate: float
        Learning rate for baselines
    - delta_bias: float
        Regularization value for baselines

"""

import numpy as np
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile
from caserec.utils.write_file import WriteFile

__author__ = "Arthur Fortes"


class MatrixFactorization(object):
    def __init__(self, train_file, test_file, prediction_file=None, steps=30, learn_rate=0.01, delta=0.015, factors=10,
                 init_mean=0.1, init_stdev=0.1, baseline=False, bias_learn_rate=0.005, delta_bias=0.002,
                 random_seed=0):
        self.train_set = ReadFile(train_file).return_information()
        self.test_set = ReadFile(test_file).return_information()
        self.prediction_file = prediction_file
        self.steps = steps
        self.learn_rate = learn_rate
        self.delta = delta
        self.factors = factors
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.baseline = baseline
        self.predictions = list()
        self.map_items = dict()
        self.map_items_index = dict()
        self.map_users = dict()
        self.map_users_index = dict()
        self.bias_learn_rate = bias_learn_rate
        self.delta_bias = delta_bias

        if random_seed != 0:
            np.random.seed(1)

        self.p = None
        self.q = None
        self.bu = None
        self.bi = None

        self.users = sorted(set(list(self.train_set['users']) + list(self.test_set['users'])))
        self.items = sorted(set(list(self.train_set['items']) + list(self.test_set['items'])))

        for i, item in enumerate(self.items):
            self.map_items.update({item: i})
            self.map_items_index.update({i: item})
        for u, user in enumerate(self.users):
            self.map_users.update({user: u})
            self.map_users_index.update({u: user})

        list_feedback = list()
        self.dict_index = dict()
        for user, item, feedback in self.train_set['list_feedback']:
            list_feedback.append((self.map_users[user], self.map_items[item], feedback))
            self.dict_index.setdefault(self.map_users[user], []).append(self.map_items[item])
        self.train_set['list_feedback'] = list_feedback
        self._create_factors()

    def _create_factors(self):
        self.p = np.random.normal(self.init_mean, self.init_stdev, (len(self.users), self.factors))
        self.q = np.random.normal(self.init_mean, self.init_stdev, (len(self.items), self.factors))

        if self.baseline:
            self.bu = np.zeros(len(self.users), np.double)
            self.bi = np.zeros(len(self.items), np.double)

    def _predict(self, u, i, cond=True):
        if self.baseline:
            rui = self.train_set["mean_rates"] + self.bu[u] + self.bi[i] + np.dot(self.p[u], self.q[i])
        else:
            rui = self.train_set['mean_rates'] + np.dot(self.p[u], self.q[i])

        if cond:
            if rui > self.train_set["max"]:
                rui = self.train_set["max"]
            elif rui < self.train_set["min"]:
                rui = self.train_set["min"]

        return rui

    def train_model(self):
        rmse_old = .0
        for epoch in range(self.steps):
            error_final = .0
            for user, item, feedback in self.train_set['list_feedback']:
                eui = feedback - self._predict(user, item, False)
                error_final += (eui ** 2.0)

                # Adjust the factors
                u_f = self.p[user]
                i_f = self.q[item]

                # Compute factor updates
                delta_u = np.subtract(np.multiply(eui, i_f), np.multiply(self.delta, u_f))
                delta_i = np.subtract(np.multiply(eui, u_f), np.multiply(self.delta, i_f))

                # apply updates
                self.p[user] += np.multiply(self.learn_rate, delta_u)
                self.q[item] += np.multiply(self.learn_rate, delta_i)

                if self.baseline:
                    self.bu[user] += self.bias_learn_rate * (eui - self.delta_bias * self.bu[user])
                    self.bi[item] += self.bias_learn_rate * (eui - self.delta_bias * self.bi[item])

            rmse_new = np.sqrt(error_final / self.train_set["ni"])
            if np.fabs(rmse_new - rmse_old) <= 0.009:
                break
            else:
                rmse_old = rmse_new

    def predict(self):
        if self.test_set is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    u, i = self.map_users[user], self.map_items[item]
                    self.predictions.append((user, item, self._predict(u, i, True)))

            if self.prediction_file is not None:
                self.predictions = sorted(self.predictions, key=lambda x: x[0])
                WriteFile(self.prediction_file, self.predictions).write_recommendation()
            return self.predictions

    def evaluate(self, predictions):
        result = RatingPredictionEvaluation()
        res = result.evaluation(predictions, self.test_set)
        print("Eval:: RMSE:" + str(res[0]) + " MAE:" + str(res[1]))

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > Matrix Factorization]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])
        print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
              " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
        print("training time:: ", timed(self.train_model), " sec")
        print("\nprediction_time:: ", timed(self.predict), " sec\n")
        self.evaluate(self.predictions)
