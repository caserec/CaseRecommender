# coding=utf-8
"""
    Matrix Factorization Collaborative Filtering Recommender
    [Rating Prediction]

    Literature:
        Koren, Yehuda and Bell, Robert and Volinsky, Chris:
        Matrix Factorization Techniques for Recommender Systems
        Journal Computer 2009.
        http://dl.acm.org/citation.cfm?id=1608614

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from caserec.utils.extra_functions import timed

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class MatrixFactorization(BaseRatingPrediction):
    def __init__(self, train_file=None, test_file=None, output_file=None, factors=10, learn_rate=0.01, epochs=30,
                 delta=0.015, init_mean=0.1, init_stdev=0.1, baseline=False, bias_learn_rate=0.005, delta_bias=0.002,
                 stop_criteria=0.009, sep='\t', output_sep='\t', random_seed=None):
        """
        Matrix Factorization for rating prediction

        Matrix factorization models map both users and items to a joint latent factor space of dimensionality f,
        such that user-item interactions are modeled as inner products in that space.

        Usage::

            >> MatrixFactorization(train, test).compute()

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

        :param baseline: Use the train data to build baselines (SVD Algorithm); else: Use only the mean
        :type baseline: bool, default False

        :param bias_learn_rate: Learning rate for baselines
        :type bias_learn_rate: float, default 0.005

        :param delta_bias: Regularization value for baselines
        :type delta_bias: float, default 0.002

        :param stop_criteria: Difference between errors for stopping criteria
        :type stop_criteria: float, default 0.001

        :param sep: Delimiter for input files
        :type sep: str, default'\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None

        """
        super(MatrixFactorization, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                                  sep=sep, output_sep=output_sep)

        self.recommender_name = 'Matrix Factorization'

        self.epochs = epochs
        self.learn_rate = learn_rate
        self.delta = delta
        self.factors = factors
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.baseline = baseline
        self.bias_learn_rate = bias_learn_rate
        self.delta_bias = delta_bias
        self.stop_criteria = stop_criteria

        if random_seed is not None:
            np.random.seed(random_seed)

        # internal vars
        self.feedback_triples = None
        self.p = None
        self.q = None
        self.bu = None
        self.bi = None

    def init_model(self):
        """
        Method to treat and initialize the model

        """
        self.feedback_triples = []

        # Map interaction with ids
        for user in self.train_set['feedback']:
            for item in self.train_set['feedback'][user]:
                self.feedback_triples.append((self.user_to_user_id[user], self.item_to_item_id[item],
                                              self.train_set['feedback'][user][item]))

        # Initialize factors
        self.create_factors()

    def fit(self):
        """
        This method performs iterations of stochastic gradient ascent over the training data.

        """

        rmse_old = .0

        for epoch in range(self.epochs):

            error_final = .0

            for user, item, feedback in self.feedback_triples:

                eui = feedback - self._predict_score(user, item, False)
                error_final += (eui ** 2.0)

                # Adjust the factors
                u_f = self.p[user]
                i_f = self.q[item]

                # Compute factor updates
                delta_u = np.subtract(np.multiply(eui, i_f), np.multiply(self.delta, u_f))
                delta_i = np.subtract(np.multiply(eui, u_f), np.multiply(self.delta, i_f))

                # apply updates
                self.p[user] += np.multiply(self.learn_rate, delta_u)
                self.q[item] += np.multiply(self.learn_rate, delta_i)

                if self.baseline:
                    self.bu[user] += self.bias_learn_rate * (eui - self.delta_bias * self.bu[user])
                    self.bi[item] += self.bias_learn_rate * (eui - self.delta_bias * self.bi[item])

            rmse_new = np.sqrt(error_final / self.train_set["number_interactions"])
            if np.fabs(rmse_new - rmse_old) <= self.stop_criteria:
                break
            else:
                rmse_old = rmse_new

    def create_factors(self):
        """
        This method create factors for users, items and bias

        """

        self.p = np.random.normal(self.init_mean, self.init_stdev, (len(self.users), self.factors))
        self.q = np.random.normal(self.init_mean, self.init_stdev, (len(self.items), self.factors))

        if self.baseline:
            self.bu = np.zeros(len(self.users), np.double)
            self.bi = np.zeros(len(self.items), np.double)

    def _predict_score(self, u, i, cond=True):
        """
        Method to predict a single score for a pair (user, item)

        :param u: User ID
        :type u: int

        :param i: Item ID
        :type i: int

        :param cond: Use max and min values of train set to limit score
        :type cond: bool, default True

        :return: Score generate for pair (user, item)
        :rtype: float

        """

        if self.baseline:
            rui = self.train_set["mean_value"] + self.bu[u] + self.bi[i] + np.dot(self.p[u], self.q[i])
        else:
            rui = self.train_set['mean_value'] + np.dot(self.p[u], self.q[i])

        if cond:
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            elif rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

        return rui

    def predict(self):
        """
        This method computes a final rating for unknown pairs (user, item)

        """

        if self.test_file is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    self.predictions.append((user, item, self._predict_score(self.user_to_user_id[user],
                                                                             self.item_to_item_id[item], True)))
        else:
            raise NotImplemented

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):
        """
        Extends compute method from BaseRatingPrediction. Method to run recommender algorithm

        :param verbose: Print recommender and database information
        :type verbose: bool, default True

        :param metrics: List of evaluation measures
        :type metrics: list, default None

        :param verbose_evaluation: Print the evaluation results
        :type verbose_evaluation: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        super(MatrixFactorization, self).compute(verbose=verbose)

        if verbose:
            self.init_model()
            print("training_time:: %4f sec" % timed(self.fit))
            if self.extra_info_header is not None:
                print(self.extra_info_header)

            print("prediction_time:: %4f sec" % timed(self.predict))

            print('\n')

        else:
            # Execute all in silence without prints
            self.init_model()
            self.fit()
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)
