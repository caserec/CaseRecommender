# coding=utf-8
"""
    gSVD++: supporting implicit feedback on recommender systems with metadata awareness
    [Rating Prediction]

    Literature:
        Marcelo Garcia Manzato. 2013. gSVD++: supporting implicit feedback on recommender systems with metadata
        awareness. In Proceedings of the 28th Annual ACM Symposium on Applied Computing (SAC '13). ACM, New York,
        NY, USA, 908-913. DOI: https://doi.org/10.1145/2480362.2480536

"""

# © 2018. Case Recommender (MIT License)

import numpy as np
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.utils.process_data import ReadFile

__author__ = 'Eduardo Fressato <eduardofressato@hotmail.com>'


class GSVDPlusPlus(MatrixFactorization):
    def __init__(self, train_file=None, test_file=None, output_file=None, metadata_file=None, factors=10,
                 learn_rate=0.01, epochs=10, delta=0.015, init_mean=0.1, init_stdev=0.1, bias_learn_rate=0.005,
                 delta_bias=0.002, stop_criteria=0.009, sep='\t', sep_metadata='\t', output_sep='\t', random_seed=None,
                 update_delta=False):
        """
        gSVD++ for rating prediction

        The gSVD++ algorithm exploits implicit feedback from users by considering not only the latent space of
        factors describing the user and item, but also the available metadata associated to the content.

        Usage::

            >> GSVDPlusPlus(train, test, metadata).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param metadata_file: File which contains the metadata. This file needs to have at least 2 columns
        (item metadata).
        :type metadata_file: str, default None

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

        super(GSVDPlusPlus, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                           factors=factors, learn_rate=learn_rate, epochs=epochs, delta=delta,
                                           init_mean=init_mean, init_stdev=init_stdev, baseline=True,
                                           bias_learn_rate=bias_learn_rate, delta_bias=delta_bias,
                                           stop_criteria=stop_criteria, sep=sep, output_sep=output_sep,
                                           random_seed=random_seed)

        self.recommender_name = 'GSVDPlusPlus'
        self.update_delta = update_delta

        self.y = None
        self.x = None
        self.n_u = None
        self.n_i = None
        self.n_g = None
        self.items_id_seen_by_user = None
        self.users_id_seen_by_item = None
        self.meta_dict_category, self.meta_list_item_category, self.meta_set_items, self.meta_dict_item_category = \
            ReadFile(metadata_file, sep_metadata).read_item_category()

        self.g_to_gid = {}
        self.gid_to_g = {}
        for i, g in enumerate(self.meta_dict_category.keys()):
            self.g_to_gid.update({g: i})
            self.gid_to_g.update({i: g})

    def init_model(self):
        """
        Method to treat and initialize the model. . Extends init_model from MatrixFactorization

        """

        super(GSVDPlusPlus, self).init_model()

        self.n_u = {}
        self.items_id_seen_by_user = {}
        self.users_id_seen_by_item = {}

        for user in self.train_set['users']:
            for item in self.train_set['items_seen_by_user'][user]:
                self.items_id_seen_by_user.setdefault(self.user_to_user_id[user], []).append(self.item_to_item_id[item])
                self.users_id_seen_by_item.setdefault(self.item_to_item_id[item], []).append(self.user_to_user_id[user])
            # |N(u)|^(-1/2)
            self.n_u[self.user_to_user_id[user]] = np.sqrt(len(self.train_set['items_seen_by_user'][user]))

        self.n_i = {}
        self.n_g = {}
        for item in self.items:
            if item in self.meta_dict_item_category:
                self.n_g[self.item_to_item_id[item]] = 1 / len(self.meta_dict_item_category[item])
            else:
                self.n_g[self.item_to_item_id[item]] = 0

            if item in self.train_set['items']:
                self.n_i[self.item_to_item_id[item]] = \
                    np.sqrt(len(self.users_id_seen_by_item[self.item_to_item_id[item]]))
            else:
                self.n_i[self.item_to_item_id[item]] = 0

    def fit(self):
        """
        This method performs iterations of stochastic gradient ascent over the training data.

        """

        rmse_old = .0
        for epoch in range(self.epochs):
            error_final = .0

            for user, item, feedback in self.feedback_triples:
                pu = self.p[user] + self.y_sum_rows(user)
                pi = self.q[item] + self.x_sum_rows(item)

                learn_rate1 = self.bias_learn_rate
                learn_rate2 = self.learn_rate
                delta1 = 0.05 * self.n_u[user]
                delta2 = 0.05 * self.n_i[item]
                delta3 = self.n_u[user]
                delta4 = self.n_i[item]

                # Calculate error
                eui = feedback - self._predict_gsvd(user, item, pu, pi, False)
                error_final += (eui ** 2.0)

                # update bu and bi
                self.bu[user] += learn_rate1 * (eui - delta1 * self.bu[user])
                self.bi[item] += learn_rate2 * (eui - delta2 * self.bi[item])

                part_2_user = (np.multiply(eui, pi) - np.multiply(delta3, self.p[user]))
                self.p[user] = self.p[user] + learn_rate2 * part_2_user

                part_2_item = (np.multiply(eui, pu) - np.multiply(delta4, self.q[item]))
                self.q[item] = self.q[item] + learn_rate2 * part_2_item

                for _g in self.meta_dict_item_category[self.item_id_to_item[item]]:
                    g = self.g_to_gid[_g]
                    delta5 = 1 / self.meta_dict_category[_g]
                    part_2 = (eui * self.n_g[item] * pu - delta5 * self.x[g])
                    self.x[g] = self.x[g] + learn_rate2 * part_2

                for j in self.items_id_seen_by_user[user]:
                    delta6 = self.n_i[j]
                    part_2 = (eui * self.n_u[user] * pi - delta6 * self.y[j])
                    self.y[j] = self.y[j] + learn_rate2 * part_2

            rmse_new = np.sqrt(error_final / self.train_set["number_interactions"])

            if np.fabs(rmse_new - rmse_old) <= self.stop_criteria:
                break
            else:
                rmse_old = rmse_new

    def create_factors(self):
        """
        This method extends create_factors from Matrix Factorization, adding y factors

        """
        super(GSVDPlusPlus, self).create_factors()
        self.y = np.random.normal(self.init_mean, self.init_stdev, (len(self.items), self.factors))
        self.x = np.random.normal(self.init_mean, self.init_stdev, (len(self.meta_dict_category), self.factors))

    def _predict_gsvd(self, u, i, pu, pi, cond=True):
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
        rui = self.train_set["mean_value"] + self.bu[u] + self.bi[i] + np.dot(pu, pi)

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

    def x_sum_rows(self, item):
        if self.item_id_to_item[item] in self.meta_dict_item_category:
            alpha = 1
        else:
            print("Error x Sum")
            alpha = 0

        sum_g = np.zeros(self.factors)
        for g in self.meta_dict_item_category[self.item_id_to_item[item]]:
            sum_g += self.x[self.g_to_gid[g]]

        g_i = len(self.meta_dict_item_category[self.item_id_to_item[item]])
        return (sum_g * alpha) / g_i

    def predict(self):
        """
        This method computes a final rating for unknown pairs (user, item)

        """

        if self.test_file is not None:
            for user in self.test_set['users']:
                pu = self.p[self.user_to_user_id[user]] + self.y_sum_rows(self.user_to_user_id[user])

                for item in self.test_set['feedback'][user]:
                    pi = self.q[self.item_to_item_id[item]] + self.x_sum_rows(self.item_to_item_id[item])
                    self.predictions.append(
                        (user, item, self._predict_gsvd(self.user_to_user_id[user],
                                                        self.item_to_item_id[item], pu, pi, True)))
        else:
            raise NotImplemented
