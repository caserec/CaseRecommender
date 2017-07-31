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
                 init_mean=0.1, init_stdev=0.1, bias_learn_rate=0.005, bias_reg=0.002, random_seed=0):
        MatrixFactorization.__init__(self, train_file=train_file, test_file=test_file, prediction_file=prediction_file,
                                     steps=steps, learn_rate=learn_rate, delta=delta, factors=factors,
                                     init_mean=init_mean, init_stdev=init_stdev, baseline=True,
                                     bias_learn_rate=bias_learn_rate, delta_bias=bias_reg, random_seed=random_seed)

        self.y = np.random.normal(self.init_mean, self.init_stdev, (len(self.items), self.factors))
        self.n_u = dict()
        # |N(u)|^(-1/2)
        for u, user in enumerate(self.users):
            self.n_u[u] = np.power(len(self.train_set["feedback"].get(user, [0])), -.5)

    def _predict_svd_plus_plus(self, u, i, sum_imp, cond=True):
        rui = self.train_set["mean_rates"] + self.bu[u] + self.bi[i] + np.dot((
            self.p[u] + self.n_u[u] * sum_imp), self.q[i])

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

                # Incorporating implicit feedback in the SVD: Sum (j E N(u)) Yj
                sum_imp = sum(self.y[self.dict_index[user]])

                # Calculate error
                eui = feedback - self._predict_svd_plus_plus(user, item, sum_imp, False)
                error_final += (eui ** 2.0)

                # Adjust the factors
                u_f = self.p[user]
                i_f = self.q[item]

                # Compute factor updates
                delta_u = np.subtract(np.multiply(eui, i_f), np.multiply(self.delta, u_f))
                delta_i = np.subtract(np.multiply(eui, (u_f + self.n_u[user] * sum_imp)), np.multiply(self.delta, i_f))

                # apply updates
                self.p[user] += np.multiply(self.learn_rate, delta_u)
                self.q[item] += np.multiply(self.learn_rate, delta_i)

                # update bu and bi
                self.bu[user] += self.bias_learn_rate * (eui - self.delta_bias * self.bu[user])
                self.bi[item] += self.bias_learn_rate * (eui - self.delta_bias * self.bi[item])

                # update y (implicit factor)
                # ∀j E N(u) :
                # yj ← yj + γ2 · (eui · |N(u)|−1/2 · qi − λ7 · yj )

                for j in self.dict_index[user]:
                    delta_y = np.subtract(eui * self.n_u[user] * self.q[item], self.delta * self.y[j])
                    self.y[j] += self.learn_rate * delta_y

            rmse_new = np.sqrt(error_final / self.train_set["ni"])

            if np.fabs(rmse_new - rmse_old) <= 0.009:
                break
            else:
                rmse_old = rmse_new

    def predict(self):
        if self.test_set is not None:
            for user in self.test_set['users']:

                # sum (j E N(u)) Yj
                try:
                    sum_imp = sum(self.y[self.dict_index[user]])
                except KeyError:
                    sum_imp = np.ones(self.factors, np.double)

                for item in self.test_set['feedback'][user]:
                    self.predictions.append((user, item, self._predict_svd_plus_plus(
                        self.map_users[user], self.map_items[item], sum_imp)))

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
