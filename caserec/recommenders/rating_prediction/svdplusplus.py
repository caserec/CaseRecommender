# coding=utf-8
"""
© 2017. Case Recommender All Rights Reserved (License GPL3)

SVD ++

Yehuda Koren: Factorization meets the neighborhood: a multifaceted collaborative filtering model, KDD 2008
http://portal.acm.org/citation.cfm?id=1401890.1401944

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
    - bias_learn_rate: float
        Learning rate for baselines
    - delta_bias: float
        Regularization value for baselines

"""

import numpy as np
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.utils.extra_functions import timed
from caserec.utils.write_file import WriteFile

__author__ = "Arthur Fortes"


class SVDPlusPlus(MatrixFactorization):
    def __init__(self, train_file, test_file, prediction_file=None, steps=30, learn_rate=0.01, delta=0.015, factors=10,
                 init_mean=0.1, init_stdev=0.1, bias_learn_rate=0.005, bias_reg=0.002):
        MatrixFactorization.__init__(self, train_file=train_file, test_file=test_file, prediction_file=prediction_file,
                                     steps=steps, learn_rate=learn_rate, delta=delta, factors=factors,
                                     init_mean=init_mean, init_stdev=init_stdev, baseline=True,
                                     bias_learn_rate=bias_learn_rate, delta_bias=bias_reg)

        self.y = self.init_mean * np.random.randn(len(self.items), self.factors) + self.init_stdev ** 2
        self.user_implicit_feedback = np.zeros((len(self.users), self.factors))

    def _predict_svd_plus_plus(self, u, i, cond=True):
        rui = self.train_set["mean_rates"] + self.bu[u] + self.bi[i] + np.dot(
            (self.p[u] + self.user_implicit_feedback[u]), self.q[i])

        if cond:
            if rui > self.train_set["max"]:
                rui = self.train_set["max"]
            elif rui < self.train_set["min"]:
                rui = self.train_set["min"]
        return rui

    def train_model(self):
        for epoch in range(self.steps):
            for user in self.train_set['feedback']:
                sqrt_iu = (np.sqrt(len(self.train_set["du"][user])))
                u = self.map_users[user]

                for item in self.train_set['feedback'][user]:
                    for item_j in self.train_set['feedback'][user]:
                        self.user_implicit_feedback[u] += (self.y[self.map_items[item_j]] / sqrt_iu)

                    feedback = self.train_set['feedback'][user][item]
                    i = self.map_items[item]
                    eui = feedback - self._predict_svd_plus_plus(u, i, False)

                    # Adjust the factors
                    u_f = self.p[u]
                    i_f = self.q[i]

                    # Compute factor updates
                    delta_u = np.subtract(np.multiply(eui, i_f), np.multiply(self.delta, u_f))
                    delta_i = np.subtract(np.multiply(eui, u_f), np.multiply(self.delta, i_f))

                    # apply updates
                    self.p[u] += np.multiply(self.learn_rate, delta_u)
                    self.q[i] += np.multiply(self.learn_rate, delta_i)

                    # update bu and bi
                    self.bu[u] += self.bias_learn_rate * (eui - self.delta_bias * self.bu[u])
                    self.bi[i] += self.bias_learn_rate * (eui - self.delta_bias * self.bi[i])

                    # update y (implicit factor)
                    for item_j in self.train_set['feedback'][user]:
                        self.y[self.map_items[item_j]] += np.multiply(
                            0.007, (np.subtract(np.multiply(eui, i_f / sqrt_iu),
                                                np.multiply(0.02, self.y[self.map_items[item_j]]))))

    def predict(self):
        if self.test_set is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    u, i = self.map_users[user], self.map_items[item]
                    self.predictions.append((user, item, self._predict_svd_plus_plus(u, i)))

            if self.prediction_file is not None:
                self.predictions = sorted(self.predictions, key=lambda x: x[0])
                WriteFile(self.prediction_file, self.predictions).write_recommendation()
            return self.predictions

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > SVD++]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])
        print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
              " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
        print("training time:: ", timed(self.train_model), " sec")
        print("\nprediction_time:: ", timed(self.predict), " sec\n")
        self.evaluate(self.predictions)
