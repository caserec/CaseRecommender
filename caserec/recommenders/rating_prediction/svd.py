# coding=utf-8
"""
    Singular Value Decomposition Based Collaborative Filtering Recommender
    [Rating Prediction]

    Literature:
        Badrul Sarwar , George Karypis , Joseph Konstan , John Riedl:
        Incremental Singular Value Decomposition Algorithms for Highly Scalable Recommender Systems
        Fifth International Conference on Computer and Information Science 2002.
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.3.7894

"""

# © 2018. Case Recommender (MIT License)

import numpy as np
from scipy.sparse.linalg import svds

from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from caserec.utils.extra_functions import timed

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class SVD(BaseRatingPrediction):
    def __init__(self, train_file=None, test_file=None, output_file=None, factors=10, sep='\t', output_sep='\t',
                 random_seed=None):
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

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None

        """
        super(SVD, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file, sep=sep,
                                  output_sep=output_sep)

        self.recommender_name = 'SVD'
        self.factors = factors

        if random_seed is not None:
            np.random.seed(random_seed)

        # internal vars
        self.feedback_triples = None
        self.prediction_matrix = None

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

        self.create_matrix()

    def fit(self):
        """
        This method performs Singular Value Decomposition over the training data.

        """

        u, s, vt = svds(self.matrix, k=self.factors)
        s_diagonal_matrix = np.diag(s)
        self.prediction_matrix = np.dot(np.dot(u, s_diagonal_matrix), vt)

    def predict_score(self, u, i, cond=True):
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

        rui = self.train_set["mean_value"] + self.prediction_matrix[u][i]

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
                    self.predictions.append((user, item, self.predict_score(self.user_to_user_id[user],
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

        super(SVD, self).compute(verbose=verbose)

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
