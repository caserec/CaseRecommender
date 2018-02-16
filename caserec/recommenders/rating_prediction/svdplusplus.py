# coding=utf-8
"""
    SVD++ Based Collaborative Filtering Recommender
    [Rating Prediction]

    Literature:
        Yehuda Koren:
        Factorization meets the neighborhood: a multifaceted collaborative filtering model
        KDD 2008
        http://portal.acm.org/citation.cfm?id=1401890.1401944

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class SVDPlusPlus(MatrixFactorization):
    def __init__(self, train_file=None, test_file=None, output_file=None, factors=10, learn_rate=0.01, epochs=10,
                 delta=0.015, init_mean=0.1, init_stdev=0.1, bias_learn_rate=0.005, delta_bias=0.002,
                 stop_criteria=0.009, sep='\t', output_sep='\t', random_seed=None, update_delta=False):
        """
        SVD++ for rating prediction

        The SVD++ algorithm, an extension of SVD taking into account implicit ratings. Just as for SVD, the parameters
        are learned using a SGD on the regularized squared error objective.

        Usage::

            >> SVDPlusPlus(train, test).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param factors: Number of latent factors per user/item
        :type factors: int, default 10

        :param learn_rate: Learning rate (alpha)
        :type learn_rate: float, default 0.05

        :param epochs: Number of epochs over the training data
        :type epochs: int, default 30

        :param delta: Regularization value
        :type delta: float, default 0.015

        :param init_mean: Mean of the normal distribution used to initialize the latent factors
        :type init_mean: float, default 0

        :param init_stdev: Standard deviation of the normal distribution used to initialize the latent factors
        :type init_stdev: float, default 0.1

        :param bias_learn_rate: Learning rate for baselines
        :type bias_learn_rate: float, default 0.005

        :param delta_bias: Regularization value for baselines
        :type delta_bias: float, default 0.002

        :param stop_criteria: Difference between errors for stopping criteria
        :type stop_criteria: float, default 0.009

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None

        """

        super(SVDPlusPlus, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                          factors=factors, learn_rate=learn_rate, epochs=epochs, delta=delta,
                                          init_mean=init_mean, init_stdev=init_stdev, baseline=True,
                                          bias_learn_rate=bias_learn_rate, delta_bias=delta_bias,
                                          stop_criteria=stop_criteria, sep=sep, output_sep=output_sep,
                                          random_seed=random_seed)

        self.recommender_name = 'SVDPlusPlus'
        self.update_delta = update_delta

        self.y = None
        self.n_u = None
        self.items_id_seen_by_user = None

    def init_model(self):
        """
        Method to treat and initialize the model. . Extends init_model from MatrixFactorization

        """

        super(SVDPlusPlus, self).init_model()

        self.n_u = {}
        self.items_id_seen_by_user = {}

        for user in self.train_set['users']:
            for item in self.train_set['items_seen_by_user'][user]:
                self.items_id_seen_by_user.setdefault(self.user_to_user_id[user], []).append(self.item_to_item_id[item])
            # |N(u)|^(-1/2)
            self.n_u[self.user_to_user_id[user]] = np.sqrt(len(self.train_set['items_seen_by_user'][user]))

    def fit(self):
        """
        This method performs iterations of stochastic gradient ascent over the training data.

        """

        rmse_old = .0
        for epoch in range(self.epochs):
            error_final = .0

            for user, item, feedback in self.feedback_triples:
                pu = self.p[user] + self.y_sum_rows(user)

                # Calculate error
                eui = feedback - self._predict_svd_plus_plus_score(user, item, pu, False)
                error_final += (eui ** 2.0)

                # update bu and bi
                self.bu[user] += self.bias_learn_rate * (eui - self.delta_bias * self.bu[user])
                self.bi[item] += self.bias_learn_rate * (eui - self.delta_bias * self.bi[item])

                # Adjust the factors
                norm_eui = eui / self.n_u[user]

                i_f = self.q[item]

                # Compute factor updates
                delta_u = np.subtract(np.multiply(eui, i_f), np.multiply(self.delta, self.p[user]))
                self.p[user] += np.multiply(self.learn_rate, delta_u)

                delta_i = np.subtract(np.multiply(eui, pu), np.multiply(self.delta, i_f))
                self.q[item] += np.multiply(self.learn_rate, delta_i)

                # update y (implicit factor)
                common_update = norm_eui * i_f

                for j in self.items_id_seen_by_user[user]:
                    delta_y = np.subtract(common_update, self.delta * self.y[j])
                    self.y[j] += self.learn_rate * delta_y

            rmse_new = np.sqrt(error_final / self.train_set["number_interactions"])

            if np.fabs(rmse_new - rmse_old) <= self.stop_criteria:
                break
            else:
                rmse_old = rmse_new

    def create_factors(self):
        """
        This method extends create_factors from Matrix Factorization, adding y factors

        """

        super(SVDPlusPlus, self).create_factors()
        self.y = np.random.normal(self.init_mean, self.init_stdev, (len(self.items), self.factors))

    def _predict_svd_plus_plus_score(self, u, i, pu, cond=True):
        """

        :param u: User ID (from self.items)
        :type u: int

        :param i: Item ID (from self.items)
        :type i: int

        :param pu: User updated vector (pu * y)
        :type pu: list or np.array

        :param cond: Use max and min values of train set to limit score
        :type cond: bool, default True

        :return: prediction for user u and item i
        :rtype: float

        """
        rui = self.train_set["mean_value"] + self.bu[u] + self.bi[i] + np.dot(pu, self.q[i])

        if cond:
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            elif rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]
        return rui

    def y_sum_rows(self, user):
        """
        Incorporating implicit feedback in the SVD: Sum (j E N(u)) Yj

        :param user: User ID
        :type user: int

        :return: Sum of y vectors for seen items of user

        """

        sum_imp = np.zeros(self.factors)
        for ui in self.items_id_seen_by_user[user]:
            sum_imp += self.y[ui]
        return sum_imp / self.n_u[user]

    def predict(self):
        """
        This method computes a final rating for unknown pairs (user, item)

        """

        if self.test_file is not None:
            for user in self.test_set['users']:
                pu = self.p[self.user_to_user_id[user]] + self.y_sum_rows(self.user_to_user_id[user])

                for item in self.test_set['feedback'][user]:
                    self.predictions.append(
                        (user, item, self._predict_svd_plus_plus_score(self.user_to_user_id[user],
                                                                       self.item_to_item_id[item], pu, True)))
        else:
            raise NotImplemented
