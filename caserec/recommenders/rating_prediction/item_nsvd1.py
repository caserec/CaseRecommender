# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

NSVD1
    Literature:
    Recommending new movies: even a few ratings are more valuable than metadata
    http://dl.acm.org/citation.cfm?id=1639731


Parameters
-----------
    - train_file: string
    - test_file: string
    - metadata_file: string
        Metadata file ; Format file:
        item \t metadata \t value\n
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

from caserec.recommenders.rating_prediction.base_nsvd1 import BaseNSVD1
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile
import numpy as np
import math

__author__ = "Arthur Fortes"


class ItemNSVD1(BaseNSVD1):
    def __init__(self, train_file, test_file, metadata_file, prediction_file=None, steps=30, learn_rate=0.01,
                 delta=0.015, factors=10, init_mean=0.1, init_stdev=0.1, alpha=0.001, batch=False, n2=10,
                 learn_rate2=0.01, delta2=0.015, space_type='\t'):
        BaseNSVD1.__init__(self, train_file, test_file, prediction_file, factors, init_mean, init_stdev, space_type)
        self.metadata = ReadFile(metadata_file, space_type=space_type).read_metadata(self.items)
        self.number_metadata = len(self.metadata["metadata"])
        self.batch = batch
        self.steps = steps
        self.learn_rate = learn_rate
        self.delta = delta
        self.alpha = alpha
        self.n2 = n2
        self.learn_rate2 = learn_rate2
        self.delta2 = delta2

        # Internal
        self.x = self.metadata['matrix']
        self.non_zero_x = list()
        self.d = list()
        for i in range(self.number_items):
            self.non_zero_x.append(list(np.where(self.x[i] != 0)[0]))
            with np.errstate(divide='ignore'):
                self.d.append(1 / np.dot(self.x[i].T, self.x[i]))

    def _update_factors(self, item, i):
        c, e = 0, 0
        try:
            for user in self.train_set['di'][item]:
                u = self.map_users[user]
                rui = self._predict(u, i)
                error = self.train_set['feedback'][user][item] - rui
                b = np.array(self.p[u])

                # update factors
                self.p[u] += self.learn_rate * (error * self.q[i] - self.delta * self.p[u])
                self.q[i] += self.learn_rate * (error * b - self.delta * self.q[i])
                self.b[u] += self.learn_rate * (error - self.delta * self.b[u])
                self.c[i] += self.learn_rate * (error - self.delta * self.c[i])
                c += 1
                e += error ** 2
        except KeyError:
            pass
        return c, e

    def train_model(self):
        for k in range(self.steps):
            rmse = 0
            count_error = 0
            for i, item in enumerate(self.items):
                self.q[i] = np.dot(self.x[i], self.w)
                a = np.array(self.q[i])
                c, e = self._update_factors(item, i)
                rmse += e
                count_error += c

                for l in self.non_zero_x[i]:
                    self.w[l] += self.d[i] * self.x[i][l] * (self.q[i] - a)
            rmse = math.sqrt(rmse / float(count_error))

            if (math.fabs(rmse - self.last_rmse)) <= self.alpha:
                break
            else:
                self.last_rmse = rmse
            print("step::", k, "RMSE::", rmse)

    def train_batch_model(self):
        for k in range(self.steps):
            rmse = 0
            count_error = 0
            self.q = np.dot(self.x, self.w)

            for i, item in enumerate(self.items):
                c, e = self._update_factors(item, i)
                rmse += e
                count_error += c

            for _ in range(self.n2):
                for i, item in enumerate(self.items):
                    e = self.q[i] - (np.dot(self.x[i], self.w))

                    for l in self.non_zero_x[i]:
                        self.w[l] += self.learn_rate2 * (self.d[i] * np.dot(self.x[i][l], e.T) -
                                                         (self.w[l] * self.delta2))

            self.q = np.dot(self.x, self.w)

            rmse = math.sqrt(rmse / float(count_error))
            if (math.fabs(rmse - self.last_rmse)) <= self.alpha:
                break
            else:
                self.last_rmse = rmse
            print("step::", k, "RMSE::", rmse)

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > Item NSVD1]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])
        print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
              " items and ", self.test_set['ni'], " interactions | sparsity ", self.test_set['sparsity'])
        print("metadata:: ", len(self.metadata['items']), " items and ", len(self.metadata['metadata']),
              " metadata and ", self.metadata['ni'], " interactions\n")
        self._create_factors()

        if self.batch:
            print("training time:: ", timed(self.train_batch_model)), " sec\n"
        else:
            print("training time:: ", timed(self.train_model)), " sec\n"

        print("\nprediction_time:: ", timed(self.predict)), " sec\n"
        self.evaluate(self.predictions)
