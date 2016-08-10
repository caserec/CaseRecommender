# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

NSVD1
    Literature:
    Improving regularized singular value decomposition for collaborative filtering
    https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf

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

from framework.recommenders.rating_prediction.base_nsvd1 import BaseNSVD1
from framework.utils.extra_functions import timed
from framework.utils.read_file import ReadFile
import numpy as np
import math

__author__ = "Arthur Fortes"


class UserNSVD1(BaseNSVD1):
    def __init__(self, train_file, test_file, metadata_file, prediction_file=None, steps=30, learn_rate=0.01,
                 delta=0.015, factors=10, init_mean=0.1, init_stdev=0.1, alpha=0.001, batch=False, n2=10,
                 learn_rate2=0.01, delta2=0.015):
        BaseNSVD1.__init__(self, train_file, test_file, prediction_file, factors, init_mean, init_stdev)
        self.metadata = ReadFile(metadata_file, space_type=" ").read_metadata(self.users)
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

    def _update_factors(self, user, u):
        c, e = 0, 0
        try:
            for item in self.train['du'][user]:
                i = self.map_items[item]
                rui = self._predict(u, i)
                error = self.train['feedback'][user][item] - rui
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
        for k in xrange(self.steps):
            rmse = 0
            count_error = 0
            for u, user in enumerate(self.users):
                self.p[u] = np.dot(self.x[u], self.w)
                a = np.array(self.p[u])
                c, e = self._update_factors(user, u)
                rmse += e
                count_error += c

                with np.errstate(divide='ignore'):
                    d = 1 / np.dot(self.x[u].T, self.x[u])

                for l in list(np.nonzero(self.x[u])[0]):
                    self.w[l] += d * self.x[u][l] * (self.p[u] - a)
            rmse = math.sqrt(rmse / float(count_error))

            if (math.fabs(rmse - self.last_rmse)) <= self.alpha:
                break
            else:
                self.last_rmse = rmse
            print k, rmse

    def train_batch_model(self):
        for k in xrange(self.steps):
            rmse = 0
            count_error = 0
            self.p = np.dot(self.x, self.w)

            for u, user in enumerate(self.users):
                c, e = self._update_factors(user, u)
                rmse += e
                count_error += c

            for _ in xrange(self.n2):
                for u, user in enumerate(self.users):
                    e = self.p[u] - (np.dot(self.x[u], self.w))
                    with np.errstate(divide='ignore'):
                        d = 1 / np.dot(self.x[u].T, self.x[u])

                    for l in list(np.nonzero(self.x[u])[0]):
                        self.w[l] += self.learn_rate2 * (d * np.dot(self.x[u][l], e.T) - np.dot(self.w[l], self.delta2))

            self.p = np.dot(self.metadata['matrix'], self.w)

            rmse = math.sqrt(rmse / float(count_error))

            if (math.fabs(rmse - self.last_rmse)) <= self.alpha:
                break
            else:
                self.last_rmse = rmse
            print k, rmse

    def execute(self):
        # methods
        print("[Case Recommender: Rating Prediction > User NSVD1]\n")
        print("training data:: " + str(len(self.train['users'])) + " users and " + str(len(
            self.train['items'])) + " items and " + str(self.train['ni']) + " interactions")
        print("test data:: " + str(len(self.test['users'])) + " users and " + str(len(self.test['items'])) +
              " items and " + str(self.test['ni']) + " interactions")
        print("metadata:: " + str(len(self.metadata['items'])) + " users and " + str(len(self.metadata['metadata'])) +
              " metadata and " + str(self.metadata['ni']) + " interactions")
        self._create_factors()

        if self.batch:
            print("training time:: " + str(timed(self.train_batch_model))) + " sec"
        else:
            print("training time:: " + str(timed(self.train_model))) + " sec"

        print("prediction_time:: " + str(timed(self.predict))) + " sec\n"
        self.evaluate(self.predictions)
