# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Matrix factorization model for item prediction (ranking) optimized using BPR.

 * BPR reduces ranking to pairwise classification.
    Literature:
        Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme:
        BPR: Bayesian Personalized Ranking from Implicit Feedback.
        UAI 2009.
        http://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf

Parameters
-----------
    - train_file: string
    - test_file: string
    - ranking_file: string
        file to write final ranking
    - factors: int
        Number of latent factors per user/item
    - learn_rate: float
        Learning rate (alpha)
    - num_interactions: int
        Number of iterations over the training data
    - num_events: int
        Number of events in each interaction
        -> default: None -> number of interactions of train file
    - predict_items_number: int
        Number of items per user in ranking
    - init_mean: float
        Mean of the normal distribution used to initialize the latent factors
    - init_stdev: float
        Standard deviation of the normal distribution used to initialize the latent factors
    - reg_u: float
        Regularization parameter for user factors
    - reg_i: float
        Regularization parameter for positive item factors
    - reg_j: float
        Regularization parameter for negative item factors
    - reg_bias: float
        Regularization parameter for the bias term
    - use_loss: bool
        Use objective function to increase learning rate

"""

import numpy as np
from framework.evaluation.item_recommendation import ItemRecommendationEvaluation
from framework.utils.extra_functions import timed
from framework.utils.read_file import ReadFile
from framework.utils.write_file import WriteFile

__author__ = "Arthur Fortes"


class BprMF(object):
    def __init__(self, train_file, test_file=None, ranking_file=None, factors=10, learn_rate=0.05, num_interactions=30,
                 num_events=None, predict_items_number=10, init_mean=0.1, init_stdev=0.1, reg_u=0.0025, reg_i=0.0025,
                 reg_j=0.00025, reg_bias=0, use_loss=True, rank_number=10):
        # external vars
        self.train_set = ReadFile(train_file).return_bprmf()
        self.test_file = test_file
        self.ranking_file = ranking_file
        self.factors = factors
        self.learn_rate = learn_rate
        self.predict_items_number = predict_items_number
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.num_interactions = num_interactions
        self.reg_bias = reg_bias
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.reg_j = reg_j
        self.use_loss = use_loss
        self.rank_number = rank_number
        self.users = self.train_set["users"]
        self.items = self.train_set["items"]
        if self.test_file is not None:
            self.test_set = ReadFile(test_file).rating_prediction()
            self.users = sorted(set(self.train_set["users"] + self.test_set["users"]))
            self.items = sorted(set(self.train_set["items"] + self.test_set["items"]))
        if num_events is None:
            self.num_events = self.train_set["number_interactions"]
        else:
            self.num_events = num_events

        # internal vars
        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.loss = None
        self.loss_sample = list()
        self.ranking = list()
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

    def _create_factors(self):
        self.p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2
        self.bias = self.init_mean * np.random.randn(self.number_items, 1) + self.init_stdev ** 2

    def _sample_triple(self):
        user = self.train_set["users"][np.random.randint(0, len(self.train_set["users"]))]
        item_i = self.train_set["feedback"][user][np.random.randint(0, len(self.train_set["feedback"][user]))]
        item_j = self.train_set["not_seen"][user][np.random.randint(0, len(self.train_set["not_seen"][user]))]
        return self.map_users[user], self.map_items[item_i], self.map_items[item_j]

    def _predict(self, user, item):
        return self.bias[item] + np.dot(self.p[user], self.q[item])

    def _update_factors(self, user, item_i, item_j):
        # Compute Difference
        eps = 1 / (1 + np.exp(self._predict(user, item_i) - self._predict(user, item_j)))

        self.bias[item_i] += self.learn_rate * (eps - self.reg_bias * self.bias[item_i])
        self.bias[item_j] += self.learn_rate * (eps - self.reg_bias * self.bias[item_j])

        # Adjust the factors
        u_f = self.p[user]
        i_f = self.q[item_i]
        j_f = self.q[item_j]

        # Compute and apply factor updates
        self.p[user] += self.learn_rate * ((i_f - j_f) * eps - self.reg_u * u_f)
        self.q[item_i] += self.learn_rate * (u_f * eps - self.reg_i * i_f)
        self.q[item_j] += self.learn_rate * (-u_f * eps - self.reg_j * j_f)

    def _compute_loss(self):
        ranking_loss = 0
        for sample in self.loss_sample:
            x_uij = self._predict(sample[0], sample[1]) - self._predict(sample[0], sample[2])
            ranking_loss += 1 / (1 + np.exp(x_uij))

        complexity = 0
        for sample in self.loss_sample:
            complexity += self.reg_u * np.linalg.norm(self.p[sample[0]]) ** 2
            complexity += self.reg_i * np.linalg.norm(self.q[sample[1]]) ** 2
            complexity += self.reg_j * np.linalg.norm(self.q[sample[2]]) ** 2
            complexity += self.reg_bias * self.bias[sample[1]] ** 2
            complexity += self.reg_bias * self.bias[sample[2]] ** 2

        return ranking_loss + 0.5 * complexity

    # Perform one iteration of stochastic gradient ascent over the training data
    # One iteration is samples number of positive entries in the training matrix times
    def train_model(self):
        if self.use_loss:
            num_sample_triples = int(np.sqrt(len(self.users)) * 100)
            for _ in xrange(num_sample_triples):
                self.loss_sample.append(self._sample_triple())
            self.loss = self._compute_loss()

        for n in xrange(self.num_interactions):
            for _ in xrange(self.num_events):
                u, i, j = self._sample_triple()
                self._update_factors(u, i, j)

            if self.use_loss:
                actual_loss = self._compute_loss()
                if actual_loss > self.loss:
                    self.learn_rate *= 0.5
                elif actual_loss < self.loss:
                    self.learn_rate *= 1.1
                self.loss = actual_loss

            print n, self.loss

    def predict(self):
        w = self.bias.T + np.dot(self.p, self.q.T)
        for u, user in enumerate(self.users):
            partial_ranking = list()
            user_list = sorted(range(len(w[u])), key=lambda k: w[u][k], reverse=True)
            for i in user_list[:100]:
                item = self.map_items_index[i]
                try:
                    if item not in self.train_set["feedback"][user]:
                        partial_ranking.append((user, item, w[u][i]))
                except KeyError:
                    partial_ranking.append((user, item, w[u][i]))
            self.ranking += partial_ranking[:self.rank_number]

        if self.ranking_file is not None:
            WriteFile(self.ranking_file, self.ranking).write_ranking_file()

    def evaluate(self):
        result = ItemRecommendationEvaluation()
        res = result.test_env(self.ranking, self.test_file)
        print("Eval:: Prec@1:" + str(res[0]) + " Prec@3:" + str(res[2]) + " Prec@5:" + str(res[4]) + " Prec@10:" +
              str(res[6]) + " Map::" + str(res[8]))

    def execute(self):
        # methods
        print("[Case Recommender: Item Recommendation > BPR MF Algorithm]\n")
        print("training data:: " + str(self.number_users) + " users and " + str(self.number_items) + " items and " +
              str(self.train_set["number_interactions"]) + " interactions")
        if self.test_file is not None:
            test_set = ReadFile(self.test_file).return_matrix()
            print("test data:: " + str(len(test_set["map_user"])) + " users and " + str(len(test_set["map_item"])) +
                  " items and " + str(test_set["number_interactions"]) + " interactions")
            del test_set
        self._create_factors()
        print("training time:: " + str(timed(self.train_model))) + " sec"
        print("prediction_time:: " + str(timed(self.predict))) + " sec\n"
        if self.test_file is not None:
            self.evaluate()
