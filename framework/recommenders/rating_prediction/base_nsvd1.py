# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Base NSVD1

Parameters
-----------
    - train_file: string
    - test_file: string
     - prediction_file: string
     - steps: int
        Number of steps over the training data
     - learn_rate: float
        Learning rate
     - delta: float
        Regularization value
     - factors: int
        Number of latent factors per user/item
     - init_mean: float
        Mean of the normal distribution used to initialize the latent factors
     - init_stdev: float
        Standard deviation of the normal distribution used to initialize the latent factors
     - alpha: float
     - batch: bool
        if True: Use batch model to train the model (default False)
     - n2: int
        Number of interactions in batch step
     - learn_rate2: float
        Learning rate in batch step
     - delta2: float
        Regularization value in Batch step

"""

from framework.evaluation.rating_prediction import RatingPredictionEvaluation
from framework.utils.read_file import ReadFile
from framework.utils.write_file import WriteFile
import numpy as np

__author__ = "Arthur Fortes"


class BaseNSVD1(object):
    def __init__(self, train_file, test_file, prediction_file=None, factors=10, init_mean=0.1,
                 init_stdev=0.1):
        self.train = ReadFile(train_file).rating_prediction()
        self.test = ReadFile(test_file).rating_prediction()
        self.prediction_file = prediction_file
        self.factors = factors
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.users = sorted(set(self.train["users"] + self.test["users"]))
        self.items = sorted(set(self.train["items"] + self.test["items"]))
        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.metadata = None
        self.number_metadata = None
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

        # internal vars
        self.x = None
        self.p = None
        self.q = None
        self.w = None
        self.b = None
        self.c = None
        self.last_rmse = 0
        self.predictions = list()

    def _create_factors(self):
        self.b = self.init_mean * np.random.randn(self.number_users, 1) + self.init_stdev ** 2
        self.c = self.init_mean * np.random.randn(self.number_items, 1) + self.init_stdev ** 2
        self.p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2
        self.w = self.init_mean * np.random.randn(self.number_metadata, self.factors) + self.init_stdev ** 2

    def _predict(self, user, item):
        return self.b[user] + self.c[item] + np.dot(self.p[user], self.q[item])

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
