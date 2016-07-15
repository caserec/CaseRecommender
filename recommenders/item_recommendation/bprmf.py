import numpy as np
from evaluation.item_recommendation import ItemRecommendationEvaluation
from utils.read_file import ReadFile
from utils.write_file import WriteFile

__author__ = "Arthur Fortes"

"""
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
        file to read final ranking
    - factors: int
        Number of latent factors per user/item
    - learn_rate: float
        Learning rate (alpha)
    - num_interactions: int
        Number of iterations over the training data
    - num_events: int
        Number of events in each interaction
        Default: None -> number of interactions of train file
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


class BprMF(object):
    def __init__(self, train_file, test_file=None, ranking_file=None, factors=10, learn_rate=0.05, num_interactions=30,
                 num_events=None, predict_items_number=10, init_mean=0.1, init_stdev=0.1, reg_u=0.0025, reg_i=0.0025,
                 reg_j=0.00025, reg_bias=0, use_loss=True):
        # external vars
        train_set = ReadFile(train_file).return_matrix()
        self.train = train_set["matrix"]
        self.map_user = train_set["map_user"]
        self.map_item = train_set["map_item"]
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
        if num_events is None:
            self.num_events = train_set["number_interactions"]
        else:
            self.num_events = num_events

        # internal vars
        self.number_users = len(self.train)
        self.number_items = len(self.train[0])
        self.loss = None
        self.loss_sample = list()
        self.ranking = list()

        # methods
        self._create_factors()
        self._sample_triple()
        self.fit()
        self.predict()
        if test_file is not None:
            self.test = test_file
            self.evaluate()

    def _create_factors(self):
        self.p = self.init_mean * np.random.randn(self.number_users, self.factors) + self.init_stdev ** 2
        self.q = self.init_mean * np.random.randn(self.number_items, self.factors) + self.init_stdev ** 2
        self.bias = self.init_mean * np.random.randn(self.number_items, 1) + self.init_stdev ** 2

    def _sample_triple(self):
        u = np.random.randint(0, len(self.train)-1)
        i = np.random.choice(np.nonzero(self.train[u])[0])
        j = np.random.choice(np.flatnonzero(self.train[u] == 0))
        return u, i, j

    #
    def _update_factors(self, user, item_i, item_j):
        # Compute Difference
        rui = self.bias[item_i] + np.dot(self.p[user], self.q[item_i])
        ruj = self.bias[item_j] + np.dot(self.p[user], self.q[item_j])

        x_uij = rui - ruj
        eps = 1 / (1 + np.exp(x_uij))

        self.bias[item_i] += self.learn_rate * (eps - self.reg_bias * self.bias[item_i])
        self.bias[item_j] += self.learn_rate * (eps - self.reg_bias * self.bias[item_j])

        # Adjust the factors
        u_f = self.p[user]
        i_f = self.q[item_i]
        j_f = self.q[item_j]

        # Compute factor updates
        delta_u = (i_f - j_f) * eps - self.reg_u * u_f
        delta_i = u_f * eps - self.reg_i * i_f
        delta_j = -u_f * eps - self.reg_j * j_f

        # Apply updates
        self.p[user] += self.learn_rate * delta_u
        self.q[item_i] += self.learn_rate * delta_i
        self.q[item_j] += self.learn_rate * delta_j

    def predict_score(self, user, item):
        return round(self.bias[item] + np.dot(self.p[user], self.q[item]), 6)

    def _compute_loss(self):
        ranking_loss = 0
        for sample in self.loss_sample:
            x_uij = self.predict_score(sample[0], sample[1]) - self.predict_score(sample[0], sample[2])
            ranking_loss += 1 / (1 + np.exp(x_uij))

        complexity = 0
        for sample in self.loss_sample:
            complexity += self.reg_u * np.power(np.linalg.norm(self.p[sample[0]]), 2)
            complexity += self.reg_i * np.power(np.linalg.norm(self.q[sample[1]]), 2)
            complexity += self.reg_j * np.power(np.linalg.norm(self.q[sample[2]]), 2)
            complexity += self.reg_bias * np.power(self.bias[sample[1]], 2)
            complexity += self.reg_bias * np.power(self.bias[sample[2]], 2)

        return ranking_loss + 0.5 * complexity

    # Perform one iteration of stochastic gradient ascent over the training data
    # One iteration is samples number of positive entries in the training matrix times
    def fit(self):
        if self.use_loss:
            num_sample_triples = int(np.sqrt(len(self.map_user)) * 100)
            for _ in xrange(num_sample_triples):
                self.loss_sample.append(self._sample_triple())
            self.loss = self._compute_loss()

        for i in xrange(self.num_interactions):
            print i
            for j in xrange(self.num_events):
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
        for user in xrange(len(self.train)):
            partial_ranking = list()
            u_list = list(np.flatnonzero(self.train[user] == 0))
            for item in u_list:
                partial_ranking.append((self.map_user[user], self.map_item[item], self.predict_score(user, item)))
            partial_ranking = sorted(partial_ranking, key=lambda x: -x[2])[:10]
            self.ranking += partial_ranking

        if self.ranking_file is not None:
            WriteFile(self.ranking_file, self.ranking).write_ranking_file()

    def evaluate(self):
        result = ItemRecommendationEvaluation()
        print result.test_env(self.ranking, self.test)
