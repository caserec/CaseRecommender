import numpy as np

__author__ = 'Arthur Fortes'


class BaseKNNRecommenders(object):
    def __init__(self, train_set, test_set):
        self.train = train_set
        self.test = test_set
        self.regBi = 10
        self.regBu = 15
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()
        self.matrix = np.zeros((len(self.train['users']), len(self.train['items'])))
        self.map_items = dict()
        self.map_users = dict()

        for item_id, item in enumerate(self.train['items']):
            self.map_items[item] = item_id

        for user_id, user in enumerate(self.train['users']):
            self.map_users[user] = user_id

        for u, user in enumerate(self.train['users']):
            for item in self.train['feedback'][user]:
                self.matrix[u][self.map_items[item]] = self.train['feedback'][user][item]

    def train_baselines(self):
        for i in xrange(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def compute_bi(self):
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

    def compute_bu(self):
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

    def compute_bui(self):
        # bui = mi + bu + bi
        for user in self.train['users']:
            for item in self.train['items']:
                self.bui.setdefault(user, {}).update({item: self.train['mean_rates'] + self.bu[user] + self.bi[item]})
        del self.bu
        del self.bi
