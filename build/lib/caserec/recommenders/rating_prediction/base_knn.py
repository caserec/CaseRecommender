# coding=utf-8
"""
    This file is base for neighborhood-based algorithms

    Used by: ItemKNN, Item Attribute KNN, UserKNN and User Attribute KNN

"""

# © 2018. Case Recommender (MIT License)

from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class BaseKNN(BaseRatingPrediction):
    def __init__(self, train_file, test_file, output_file=None, reg_bi=10, reg_bu=15, similarity_metric='cosine',
                 sep='\t', output_sep='\t'):
        """
        This class is base for all neighborhood-based algorithms.

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param reg_bi: Regularization factor for items
        :type reg_bi: int, default 10

        :param reg_bu: Regularization factor for users
        :type reg_bu: int, default 15

        :param similarity_metric:
        :type similarity_metric: str, default cosine

        :param sep: Delimiter for input files
        :type sep: str, default'\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        """
        super(BaseKNN, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                      similarity_metric=similarity_metric, sep=sep, output_sep=output_sep)

        self.reg_bi = reg_bi
        self.reg_bu = reg_bu

        # internal vars
        self.number_users = None
        self.number_items = None
        self.bu = {}
        self.bi = {}
        self.bui = {}

    def init_model(self):
        """
        Method to treat and initialize the model. Create a matrix user x item

        """

        self.number_users = len(self.users)
        self.number_items = len(self.items)

        self.create_matrix()

    def train_baselines(self):
        """
        Method to train baselines for each pair user, item

        """

        self.bu = {}
        self.bi = {}
        self.bui = {}

        for i in range(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def compute_bi(self):
        """
        Method to compute bi values

        bi = (rui - mi - bu) / (regBi + number of interactions)

        """

        self.bi = dict()

        for item in self.items:
            count = 0

            for user in self.train_set['users_viewed_item'].get(item, []):
                self.bi[item] = self.bi.get(item, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                self.train_set['mean_value'] - self.bu.get(user, 0)
                count += 1

            if count > 1:
                self.bi[item] = float(self.bi[item]) / float(self.reg_bi + count)
            elif count == 0:
                self.bi[item] = self.train_set['mean_value']

    def compute_bu(self):
        """
        Method to compute bu values

        bu = (rui - mi - bi) / (regBu + number of interactions)

        """

        self.bu = dict()
        for user in self.users:
            count = 0

            for item in self.train_set['items_seen_by_user'].get(user, []):
                self.bu[user] = self.bu.get(user, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                self.train_set['mean_value'] - self.bi.get(item, 0)
                count += 1

            if count > 1:
                self.bu[user] = float(self.bu[user]) / float(self.reg_bu + count)
            elif count == 0:
                self.bu[user] = self.train_set['mean_value']

    def compute_bui(self):
        """
        Method to compute bui values

        bui = mi + bu + bi
        """

        for user in self.users:
            for item in self.items:
                self.bui.setdefault(user, {}).update(
                    {item: self.train_set['mean_value'] + self.bu.get(user, 0) + self.bi.get(item, 0)})

        del self.bu
        del self.bi
