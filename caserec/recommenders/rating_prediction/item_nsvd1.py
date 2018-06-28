# coding=utf-8
"""
    ItemNSVD1 Collaborative Filtering Recommender
    [Rating Prediction]


    Literature:
    István Pilászy and 	Domonkos Tikk:
    Recommending new movies: even a few ratings are more valuable than metadata
    RecSys 2009
    https://dl.acm.org/citation.cfm?id=1639731

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

from caserec.recommenders.rating_prediction.base_nsvd1 import BaseNSVD1
from caserec.utils.extra_functions import timed
from caserec.utils.process_data import ReadFile

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class ItemNSVD1(BaseNSVD1):
    def __init__(self, train_file=None, test_file=None, metadata_file=None, output_file=None, epochs=30,
                 learn_rate=0.01, delta=0.015, factors=10, init_mean=0, init_stdev=0.1, stop_criteria=0.001,
                 batch=False, n2=10, learn_rate2=0.01, delta2=0.015, sep='\t', output_sep='\t', metadata_sep='\t',
                 metadata_as_binary=False, random_seed=None):
        """
        ItemNSVD1 for rating prediction

        Usage::

            >> ItemNSVD1(train, test, metadata_file='user_metadata.dat').compute()
            >> ItemNSVD1(train, test, metadata_file='user_metadata.dat', batch=True).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param metadata_file: File which contains the metadata set. This file needs to have at least 2 columns
        (user metadata).
        :type metadata_file: str

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param epochs: Number of epochs over the training data
        :type epochs: int, default 10

        :param learn_rate: Learning rate (alpha)
        :type learn_rate: float, default 0.05

        :param delta: Regularization value
        :type delta: float, default 0.015

        :param factors: Number of latent factors per user/item
        :type factors: int, default 10

        :param init_mean: Mean of the normal distribution used to initialize the latent factors
        :type init_mean: float, default 0

        :param init_stdev: Standard deviation of the normal distribution used to initialize the latent factors
        :type init_stdev: float, default 0.1

        :param stop_criteria: Difference between errors for stopping criteria
        :type stop_criteria: float, default 0.001

        :param batch: Tf True, use batch model to train the model
        :type batch: bool, default False

        :param n2: Number of interactions in batch step
        :type n2: int, default 10

        :param learn_rate2: Learning rate in batch step
        :type learn_rate2: float, default 0.01

        :param delta2: Regularization value in Batch step
        :type delta2: float, default 0.015

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param metadata_sep: Delimiter for similarity or metadata file
        :type metadata_sep: str, default '\t'

        :param metadata_as_binary: f True, the explicit value will be transform to binary
        :type metadata_as_binary: bool, default False

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None

        """

        super(ItemNSVD1, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                        factors=factors, init_mean=init_mean, init_stdev=init_stdev, sep=sep,
                                        output_sep=output_sep, random_seed=random_seed)

        self.recommender_name = 'ItemNSVD1'

        self.metadata_file = metadata_file
        self.batch = batch
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.delta = delta
        self.stop_criteria = stop_criteria
        self.n2 = n2
        self.learn_rate2 = learn_rate2
        self.delta2 = delta2
        self.metadata_sep = metadata_sep
        self.metadata_as_binary = metadata_as_binary

        # internal vars
        self.x = None
        self.non_zero_x = None
        self.d = None

    def init_model(self):
        """
        Method to treat and initialize the model. Extends init_model from BaseNSVD1

        """

        super(ItemNSVD1, self).init_model()

        self.non_zero_x = []
        self.d = []

        self.metadata = ReadFile(self.metadata_file, sep=self.metadata_sep, as_binary=self.metadata_as_binary
                                 ).read_metadata_or_similarity()

        # create metadata matrix (user x metadata)
        self.x = np.zeros((self.number_items, len(self.metadata['col_2'])))

        meta_to_meta_id = {}
        for m, data in enumerate(self.metadata['col_2']):
            meta_to_meta_id[data] = m

        for item in self.metadata['col_1']:
            for m in self.metadata['dict'][item]:
                self.x[self.item_to_item_id[item], meta_to_meta_id[m]] = self.metadata['dict'][item][m]

        # create header info for metadata
        sparsity = (1 - (self.metadata['number_interactions'] /
                         (len(self.metadata['col_1']) * len(self.metadata['col_2'])))) * 100

        self.extra_info_header = ">> metadata:: %d items and %d metadata (%d interactions) | sparsity:: %.2f%%" % \
                                 (len(self.metadata['col_1']), len(self.metadata['col_2']),
                                  self.metadata['number_interactions'], sparsity)

        self.number_metadata = len(self.metadata['col_2'])

        for i in range(self.number_items):
            self.non_zero_x.append(list(np.where(self.x[i] != 0)[0]))
            with np.errstate(divide='ignore'):
                self.d.append(1 / np.dot(self.x[i].T, self.x[i]))

        # Create Factors
        self.create_factors()

    def fit(self):
        """
        This method performs iterations of stochastic gradient ascent over the training data.

        """

        for k in range(self.epochs):
            rmse = 0
            count_error = 0

            if self.batch:
                self.q = np.dot(self.x, self.w)

                for i, item in enumerate(self.items):
                    c, e = self.update_factors(item, i)
                    rmse += e
                    count_error += c

                for _ in range(self.n2):
                    for i, item in enumerate(self.items):
                        e = self.q[i] - (np.dot(self.x[i], self.w))

                        for l in self.non_zero_x[i]:
                            self.w[l] += self.learn_rate2 * (self.d[i] * np.dot(self.x[i][l], e.T) -
                                                             (self.w[l] * self.delta2))

                self.q = np.dot(self.x, self.w)
            else:
                for i, item in enumerate(self.items):
                    self.q[i] = np.dot(self.x[i], self.w)
                    a = np.array(self.q[i])
                    c, e = self.update_factors(item, i)
                    rmse += e
                    count_error += c

                    for l in self.non_zero_x[i]:
                        self.w[l] += self.d[i] * self.x[i][l] * (self.q[i] - a)

            rmse = np.sqrt(rmse / float(count_error))

            if (np.fabs(rmse - self.last_rmse)) <= self.stop_criteria:
                break
            else:
                self.last_rmse = rmse

    def update_factors(self, item, i):
        c, e = 0, 0

        for user in self.train_set['users_viewed_item'].get(item, []):
            u = self.user_to_user_id[user]
            rui = self._predict(u, i)
            error = self.train_set['feedback'][user][item] - rui

            b = np.array(self.p[u])

            # update factors
            self.p[u] += self.learn_rate * (error * self.q[i] - self.delta * self.p[u])
            self.q[i] += self.learn_rate * (error * b - self.delta * self.q[i])
            self.b[u] += self.learn_rate * (error - self.delta * self.b[u])
            self.c[i] += self.learn_rate * (error - self.delta * self.c[i])
            c += 1
            e += error ** 2

        return c, e

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

        super(ItemNSVD1, self).compute(verbose=verbose)

        if verbose:
            self.init_model()
            if self.extra_info_header is not None:
                print(self.extra_info_header)
            print("training_time:: %4f sec" % timed(self.fit))
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
