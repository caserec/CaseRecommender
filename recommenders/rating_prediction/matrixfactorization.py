import numpy as np

__author__ = "Arthur Fortes"

"""
Matrix Factorization Based Collaborative Filtering Recommender

"""


class MatrixFactorization(object):
    def __init__(self, train_set, test_set, steps=30, gamma=0.01, delta=0.015, factors=10, init_mean=0.1,
                 init_stdev=0.1, baseline=False):
        self.train = train_set
        self.test = test_set
        self.steps = steps
        self.gamma = gamma
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
        if self.baseline:
            self._train_baselines()
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
                    self.p[u] += self.gamma * delta_u
                    self.q[self.map_items[item]] += self.gamma * delta_i

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
            return self.predictions
