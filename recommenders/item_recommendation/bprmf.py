import random

import numpy as np

"""

BPR MF Algorithm

"""


class BprMF(object):
    def __init__(self, input_file, output_file, factors=10, learn_rate=0.05):
        # external vars
        self.input_file = input_file
        self.output_file = output_file
        self.factors = factors
        self.num_interactions = 30
        self.num_events = 0
        self.reg_bias = 0
        self.reg_u = 0.0025
        self.reg_i = 0.0025
        self.reg_j = 0.00025
        self.learn_rate = learn_rate

        # internal vars
        self.list_users = set()
        self.list_items = set()
        self.dict_user = dict()
        self.dict_user_non_seen_items = dict()
        self.users_factors = dict()
        self.items_factors = dict()
        self.bias = dict()

        # called methods
        self.read_file()
        self.create_factors()
        self.train()
        self.predict()

    def read_file(self):
        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    self.num_events += 1
                    inline = line.split("\t")
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.list_users.add(user)
                    self.list_items.add(item)
                    self.dict_user.setdefault(user, {}).update({item: feedback})

        self.list_users = sorted(self.list_users)
        self.list_items = sorted(self.list_items)

        for user in self.list_users:
            self.dict_user_non_seen_items.setdefault(user,
                                                     list(set(self.list_items) - set(self.dict_user[user].keys())))

    def create_factors(self):
        for user in self.list_users:
            self.users_factors.setdefault(user, [random.uniform(0, 1) for _ in xrange(self.factors)])
        for item in self.list_items:
            self.items_factors.setdefault(item, [random.uniform(0, 1) for _ in xrange(self.factors)])
            self.bias.setdefault(item, random.uniform(0, 1))

    def sample_triple(self):
        u = random.choice(self.list_users)
        i = random.choice(self.dict_user[u].keys())
        j = random.choice(self.dict_user_non_seen_items[u])

        return u, i, j

    def update_factors(self, user, item_i, item_j):
        # x_uij = (bias_i + pu*qi) -  (bias_j + pu*qj)
        rui = sum(np.array(self.users_factors[user]) * np.array(self.items_factors[item_i]))
        ruj = sum(np.array(self.users_factors[user]) * np.array(self.items_factors[item_j]))
        x_uij = self.bias[item_i] - self.bias[item_j] + (rui - ruj)

        eps = 1.0 / (1.0 + np.exp(x_uij))

        self.bias[item_i] += self.learn_rate * (eps - self.reg_bias * self.bias[item_i])
        self.bias[item_j] += self.learn_rate * (eps - self.reg_bias * self.bias[item_j])

        for i in xrange(self.factors):
            w_uf = self.users_factors[user][i]
            h_if = self.items_factors[item_i][i]
            h_jf = self.items_factors[item_j][i]

            update = (h_if - h_jf) * eps - self.reg_u * w_uf
            self.users_factors[user][i] = w_uf + self.learn_rate * update

            update = w_uf * eps - self.reg_i * h_if
            self.items_factors[item_i][i] = h_if + self.learn_rate * update

            update = -w_uf * eps - self.reg_j * h_jf
            self.items_factors[item_j][i] = h_jf + self.learn_rate * update

    def train(self):
        for i in xrange(self.num_interactions):
            print i
            for j in xrange(self.num_events):
                user, item_i, item_j = self.sample_triple()
                self.update_factors(user, item_i, item_j)

    def predict(self):
        ranking_final = list()
        for user in self.list_users:
            list_ranking = list()
            for item in self.dict_user_non_seen_items[user]:
                score = self.bias[item] + sum(
                    np.array(self.users_factors[user]) * np.array(self.items_factors[item]))
                list_ranking.append([item, score])
            list_ranking = sorted(list_ranking, key=lambda x: x[1], reverse=True)
            ranking_final.append(list_ranking[:10])

        with open(self.output_file, "w") as infile_write:
            for u, user in enumerate(self.list_users):
                for pair in ranking_final[u][:10]:
                    infile_write.write(str(user) + "\t" + str(pair[0]) + "\t" + str(pair[1]) + "\n")
