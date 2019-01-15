# coding=utf-8
"""
    This class is base for NSVD1 algorithms.

    Used by: ItemNSVD1, and UserNSVD1

    Literature:
    István Pilászy and 	Domonkos Tikk:
    Recommending new movies: even a few ratings are more valuable than metadata
    RecSys 2009
    https://dl.acm.org/citation.cfm?id=1639731

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class BaseNSVD1(BaseRatingPrediction):
    def __init__(self, train_file, test_file, output_file=None, factors=10, init_mean=0, init_stdev=0.1,
                 sep='\t', output_sep='\t', random_seed=None):
        """
        This class is base for all NSVD1 algorithms.

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

        :param init_mean: Mean of the normal distribution used to initialize the latent factors
        :type init_mean: float, default 0

        :param init_stdev: Standard deviation of the normal distribution used to initialize the latent factors
        :type init_stdev: float, default 0.1

        :param sep: Delimiter for input files
        :type sep: str, default'\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None

        """
        super(BaseNSVD1, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file, sep=sep,
                                        output_sep=output_sep)

        self.factors = factors
        self.init_mean = init_mean
        self.init_stdev = init_stdev

        if random_seed is not None:
            np.random.seed(random_seed)

        # internal vars
        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.item_to_item_id = {}
        self.item_id_to_item = {}
        self.user_to_user_id = {}
        self.user_id_to_user = {}
        self.x = None
        self.p = None
        self.q = None
        self.w = None
        self.b = None
        self.c = None
        self.metadata = None
        self.number_metadata = None

        self.last_rmse = 0
        self.predictions = []

    def init_model(self):
        """
        Method to treat and initialize the model

        """

        # Map items and users with their respective ids and upgrade unobserved items with test set samples
        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})

    def create_factors(self):
        self.b = np.random.normal(self.init_mean, self.init_stdev, self.number_users)
        self.c = np.random.normal(self.init_mean, self.init_stdev, self.number_items)
        self.p = np.random.normal(self.init_mean, self.init_stdev, (self.number_users, self.factors))
        self.q = np.random.normal(self.init_mean, self.init_stdev, (self.number_items, self.factors))
        self.w = np.random.normal(self.init_mean, self.init_stdev, (self.number_metadata, self.factors))

    def _predict(self, user, item, cond=True):
        rui = self.b[user] + self.c[item] + np.dot(self.p[user], self.q[item])

        if cond:
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

        return rui

    def predict(self):
        """
        This method computes a final rating for unknown pairs (user, item)

        """

        if self.test_file is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    rui = self._predict(self.user_to_user_id[user], self.item_to_item_id[item])
                    self.predictions.append((user, item, rui))
        else:
            raise NotImplemented
