import numpy as np
import time

__author__ = "Arthur Fortes"


class MatrixFactorization(object):
    def __init__(self, train_set, test_set, steps=30, gamma=0.01, delta=0.015, factors=10, init_mean=0.1,
                 init_stdev=0.1):
        self.train = train_set
        self.test = test_set
        self.steps = steps
        self.gamma = gamma
        self.delta = delta
        self.factors = factors
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.predictions = list()

        self.number_users = len(self.train["users"])
        self.number_items = len(self.train["items"])
        self.map_items = dict()
        self.map_users = dict()
        for i, item in enumerate(self.train["items"]):
            self.map_items.update({item: i})
        for u, user in enumerate(self.train["users"]):
            self.map_users.update({user: u})
        self.p = None
        self.q = None
        self.final_matrix = None

        # methods
        self.create_factors()
        starting_point = time.time()
        self.train_mf()
        elapsed_time = time.time() - starting_point
        print("- Training time: " + str(elapsed_time) + " second(s)")
        starting_point = time.time()
        self.predict()
        elapsed_time = time.time() - starting_point
        print("- Prediction time: " + str(elapsed_time) + " second(s)")

    def create_factors(self):
        self.p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2

    def _predict(self, user, item, cond=True):
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
                    self.p[u] += self.gamma * delta_u
                    self.q[self.map_items[item]] += self.gamma * delta_i

            rmse = np.sqrt(error_final / self.train["ni"])
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
