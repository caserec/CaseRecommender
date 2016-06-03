import numpy as np
from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'


class BaseRatingPrediction(object):
    def __init__(self, train_file, test_file="", space_type="\t"):
        self.space_type = space_type
        self.train_file = train_file
        self.test_file = test_file
        self.train_feedback, self.train_users, self.train_items, self.train_du, \
            self.train_di, self.train_mean_rates = ReadFile(self.train_file).rating_prediction()
        self.test_feedback, self.test_users, self.test_items, self.test_du, \
            self.test_di, self.test_mean_rates = ReadFile(self.test_file).rating_prediction()

        self.regBi = 10
        self.regBu = 15
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()
        self.matrix = np.zeros((len(self.train_users), len(self.train_items)))
        self.map_items = dict()
        self.map_users = dict()

        for item_id, item in enumerate(self.train_items):
            self.map_items[item] = item_id

        for user_id, user in enumerate(self.train_users):
            self.map_users[user] = user_id

        for u, user in enumerate(self.train_users):
            for item in self.train_feedback[user]:
                self.matrix[u][self.map_items[item]] = self.train_feedback[user][item]

    def train_baselines(self):
        for i in xrange(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def compute_bi(self):
        # bi = (rui - mi - bu) / (regBi + number of interactions)
        self.bi = dict()

        for item in self.train_items:
            cont = 0
            for user in self.train_di[item]:
                self.bi[item] = self.bi.get(item, 0) + float(self.train_feedback[user][item]) - \
                                self.train_mean_rates - self.bu.get(user, 0)
                cont += 1
            if cont > 1:
                self.bi[item] = float(self.bi[item]) / float(self.regBi + cont)

    def compute_bu(self):
        # bu = (rui - mi - bi) / (regBu + number of interactions)
        self.bu = dict()
        for user in self.train_users:
            cont = 0
            for item in self.train_du[user]:
                self.bu[user] = self.bu.get(user, 0) + float(self.train_feedback[user][item]) - \
                                self.train_mean_rates - self.bi.get(item, 0)
                cont += 1
            if cont > 1:
                self.bu[user] = float(self.bu[user]) / float(self.regBu + cont)

    def compute_bui(self):
        # bui = mi + bu + bi
        for user in self.train_users:
            for item in self.train_items:
                self.bui.setdefault(user, {}).update({item: self.train_mean_rates + self.bu[user] + self.bi[item]})
        del self.bu
        del self.bi
