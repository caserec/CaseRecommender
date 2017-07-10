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
                 init_mean=0.1, init_stdev=0.1, baseline=False, bias_learn_rate=0.005, delta_bias=0.002):
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

        self.p = None
        self.q = None
        self.bu = None
        self.bi = None

        self.users = sorted(set(list(self.train_set['users']) + list(self.test_set['users'])))
        self.items = sorted(set(list(self.train_set['items']) + list(self.test_set['items'])))
        self._create_factors()

        for i, item in enumerate(self.items):
            self.map_items.update({item: i})
            self.map_items_index.update({i: item})
        for u, user in enumerate(self.users):
            self.map_users.update({user: u})
            self.map_users_index.update({u: user})

    def _create_factors(self):
        self.p = self.init_mean * np.random.randn(len(self.users), self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(len(self.items), self.factors) + self.init_stdev ** 2
        self.bi = self.init_mean * np.random.randn(len(self.items), 1) + self.init_stdev ** 2
        self.bu = self.init_mean * np.random.randn(len(self.users), 1) + self.init_stdev ** 2

    def _predict(self, u, i, cond=True):
        if self.baseline:
            rui = self.train_set["mean_rates"] + self.bu[u] + self.bi[i] + np.dot(self.p[u], self.q[i])
        else:
            rui = self.train_set["mean_rates"] + np.dot(self.p[u], self.q[i])

        if cond:
            if rui > self.train_set["max"]:
                rui = self.train_set["max"]
            elif rui < self.train_set["min"]:
                rui = self.train_set["min"]
        return rui

    def train_model(self):
        for step in range(self.steps):
            error_final = 0.0
            for user in self.train_set['users']:
                u = self.map_users[user]
                for item in self.train_set['feedback'][user]:
                    i = self.map_items[item]
                    eui = self.train_set['feedback'][user][item] - self._predict(u, i, cond=False)
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

                    # if baseline = True, update bu and bi
                    if self.baseline:
                        self.bu[u] += self.bias_learn_rate * (eui - self.delta_bias * self.bu[u])
                        self.bi[i] += self.bias_learn_rate * (eui - self.delta_bias  * self.bi[i])

            # print error in each step
            # rmse = np.sqrt(error_final / self.train_set["ni"])
            # print("step::", step, "RMSE::", rmse)

    def predict(self):
        if self.test_set is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    u, i = self.map_users[user], self.map_items[item]
                    self.predictions.append((user, item, self._predict(u, i)))

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
