# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Ensemble BPR Learning - Optimizing ensemble rankings

    Literature:
        Arthur Fortes da Costa and Marcelo G. Manzato
        Exploiting multimodal interactions in recommender systems with ensemble algorithms
        Journal Information Systems 2016.
        http://www.sciencedirect.com/science/article/pii/S0306437915300818

Parameters
-----------

"""

import numpy as np

from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from caserec.utils.read_file import ReadFile
from caserec.utils.write_file import WriteFile

__author__ = 'Arthur Fortes'


class EnsembleLearningBPR(object):
    def __init__(self, list_train_files, list_rankings_files, ranking_file=None, test_file=None, rank_number=10,
                 space_type='\t', init_mean=0.1, init_stdev=0.1, factors=10, num_interactions=30, learn_rate=0.05,
                 reg_u=0.0025, reg_i=0.0025, reg_j=0.00025, reg_bias=0, use_loss=True):
        self.list_train_files = list_train_files
        self.list_rankings_files = list_rankings_files
        self.ranking_file = ranking_file
        self.test_file = test_file
        self.rank_number = rank_number
        self.space_type = space_type
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.factors = factors
        self.num_interactions = num_interactions
        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.reg_j = reg_j
        self.use_loss = use_loss

        # external vars
        self.train_info = None
        self.rankings_info = None
        self.list_users = None
        self.list_items = None
        self.number_users = None
        self.number_items = None
        self.num_events = None
        self.ir = None
        self.rf = dict()
        self.map_user = dict()
        self.map_item = dict()
        self.loss = None
        self.loss_sample = list()
        self.ranking = list()

    def _create_factors(self):
        p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2
        bias = self.init_mean * np.random.randn(self.number_items, 1) + self.init_stdev ** 2
        beta = self.init_mean * np.random.randn(self.number_users, 1) + self.init_stdev ** 2
        return p, q, bias, beta

    def read_training_data(self):
        self.train_info, self.list_users, self.list_items, self.num_events, _ = ReadFile(
            self.list_train_files).ensemble_test()
        self.number_users = len(self.list_users)
        self.number_items = len(self.list_items)
        # remove

        self.num_events = 5000

        for u, user in enumerate(self.list_users):
            self.map_user[user] = u
        for i, item in enumerate(self.list_items):
            self.map_item[item] = i

        for r in range(len(self.list_train_files)):
            p, q, bias, beta = self._create_factors()
            self.rf[r] = {"p": p, "q": q, "bias": bias, "beta": beta}

    def _compute_loss(self):
        ranking_loss = 0
        for sample in self.loss_sample:
            x_uij = self._predict_score(sample[0], sample[1]) - self._predict_score(sample[0], sample[2])
            ranking_loss += 1 / (1 + np.exp(x_uij))

        complexity = 0
        for r in range(len(self.list_train_files)):
            for sample in self.loss_sample:
                u, i, j = sample[0], sample[1], sample[2]
                u, i, j = self.map_user[u], self.map_item[i], self.map_item[j]
                complexity += self.reg_u * np.power(np.linalg.norm(self.rf[r]["p"][u]), 2)
                complexity += self.reg_i * np.power(np.linalg.norm(self.rf[r]["q"][i]), 2)
                complexity += self.reg_j * np.power(np.linalg.norm(self.rf[r]["q"][j]), 2)
                complexity += self.reg_bias * np.power(self.rf[r]["bias"][i], 2)
                complexity += self.reg_bias * np.power(self.rf[r]["bias"][j], 2)

        return (ranking_loss/2.0) + 0.5 * (complexity/2.0)

    def read_rankings(self):
        self.rankings_info, _, _, _, self.ir = ReadFile(
            self.list_rankings_files).ensemble_test()

    def _sample_triple(self):
        u = np.random.choice(self.list_users)
        i = np.random.choice(self.train_info[u]["i"])
        j = np.random.choice(self.rankings_info[u]["i"])
        return u, i, j

    def _predict_score(self, user, item):
        u, i = self.map_user[user], self.map_item[item]
        rating = 0
        for r in range(len(self.list_train_files)):
            rating += self.rf[r]["bias"][i] + np.dot(self.rf[r]["p"][u], self.rf[r]["q"][i])
        return rating

    def _update_factors(self, user, item_i, item_j):
        u, i, j = self.map_user[user], self.map_item[item_i], self.map_item[item_j]
        rui = 0
        ruj = 0
        x_uij = 0

        for r in range(len(self.list_rankings_files)):
            try:
                self.ir[r][user][item_i] = 0
                rui += self.rf[r]["bias"][i] + np.dot(self.rf[r]["p"][u], self.rf[r]["q"][i])
                ruj += self.rf[r]["bias"][j] + np.dot(self.rf[r]["p"][u], self.rf[r]["q"][j])
            except KeyError:
                pass

            # x_uij += self.rf[r]["beta"][u] * (rui - ruj)
            x_uij += (rui - ruj)

        eps = 1 / (1 + np.exp(x_uij))

        for r in range(len(self.list_rankings_files)):
            try:
                self.ir[r][user][item_i] = 0
                self.rf[r]["bias"][i] += self.learn_rate * (eps - self.reg_bias * self.rf[r]["bias"][i])
                self.rf[r]["bias"][j] += self.learn_rate * (eps - self.reg_bias * self.rf[r]["bias"][j])

                # Adjust the factors
                u_f = self.rf[r]["p"][u]
                i_f = self.rf[r]["q"][i]
                j_f = self.rf[r]["q"][j]

                # Compute factor updates
                delta_u = (i_f - j_f) * eps - self.reg_u * u_f
                delta_i = u_f * eps - self.reg_i * i_f
                delta_j = -u_f * eps - self.reg_j * j_f

                self.rf[r]["p"][u] += self.learn_rate * delta_u
                self.rf[r]["q"][i] += self.learn_rate * delta_i
                self.rf[r]["q"][j] += self.learn_rate * delta_j

            except KeyError:
                pass

    def train_model(self):
        if self.use_loss:
            num_sample_triples = int(np.sqrt(len(self.map_user)) * 100)
            for _ in range(num_sample_triples):
                self.loss_sample.append(self._sample_triple())
            self.loss = self._compute_loss()

        for i in range(self.num_interactions):
            for j in range(self.num_events):
                user, item_i, item_j = self._sample_triple()
                self._update_factors(user, item_i, item_j)

            if self.use_loss:
                actual_loss = self._compute_loss()
                if actual_loss > self.loss:
                    self.learn_rate *= 0.5
                elif actual_loss < self.loss:
                    self.learn_rate *= 1.1
                self.loss = actual_loss

    def predict(self):
        for user in self.rankings_info:
            partial_ranking = list()
            for item in self.rankings_info[user]["i"]:
                partial_ranking.append((self.map_user[user], self.map_item[item], self._predict_score(user, item)))
            partial_ranking = sorted(partial_ranking, key=lambda x: -x[2])[:self.rank_number]
            self.ranking += partial_ranking

        if self.ranking_file is not None:
            WriteFile(self.ranking_file, self.ranking).write_recommendation()

    def evaluate(self, measures):
        res = ItemRecommendationEvaluation().evaluation_ranking(self.ranking, self.test_file)
        evaluation = 'Eval:: '
        for measure in measures:
            evaluation += measure + ': ' + str(res[measure]) + ' '
        print(evaluation)

    def execute(self, measures=('Prec@5', 'Prec@10', 'NDCG@5', 'NDCG@10', 'MAP@5', 'MAP@10')):
        # methods
        print("[Case Recommender: Item Recommendation > Ensemble BPR Learning Algorithm]\n")
        self.read_training_data()
        self.read_rankings()
        self.train_model()
        self.predict()
        if self.test_file is not None:
            self.evaluate(measures)
