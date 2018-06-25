# coding=utf-8
""""
    Group Based Recommender
    [Item Recommendation (Ranking)]

    Literature:
        Arthur Fortes da Costa, Marcelo G. Manzato, Ricardo J. G. B. Campello:
        Group-based Collaborative Filtering Supported by Multiple Users' Feedback to Improve Personalized Ranking.
        WebMedia 2016.
        https://dl.acm.org/citation.cfm?doid=2976796.2976852

"""

# © 2018. Case Recommender (MIT License)

from scipy.spatial.distance import squareform, pdist
import numpy as np
import os

from caserec.clustering.kmedoids import kmedoids
from caserec.recommenders.item_recommendation.base_item_recommendation import BaseItemRecommendation
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.utils.process_data import ReadFile, WriteFile
from caserec.recommenders.item_recommendation.bprmf import BprMF

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class GroupBasedRecommender(BaseItemRecommendation):
    def __init__(self, train_files=None, test_file=None, output_file=None, similarity_metric="cosine", rank_length=10,
                 k_groups=3, recommender='UserKNN', as_binary=False, sep='\t', output_sep='\t', max_int_kmedoids=1000,
                 parser='', user_weights=False):
        """
        Group-Based for Item Recommendation

        This algorithm predicts a rank for each user using a co-clustering algorithm

        Usage::

            >> GroupBasedRecommender([train_history], test).compute()
            >> GroupBasedRecommender([train_history, train_rating], test, as_binary=True).compute()

        :param train_files: List of train files
        :type train_files: list

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param similarity_metric: Pairwise metric to compute the similarity between the users. Reference about
        distances: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
        :type similarity_metric: str, default cosine

        :param as_binary: If True, the explicit feedback will be transform to binary
        :type as_binary: bool, default False

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        """

        super(GroupBasedRecommender, self).__init__(train_file='', test_file=test_file,
                                                    output_file=output_file, as_binary=as_binary,
                                                    rank_length=rank_length, similarity_metric=similarity_metric,
                                                    sep=sep, output_sep=output_sep)

        self.recommender_name = 'Group Based Recommender Algorithm'
        self.train_files = train_files
        self.k_groups = k_groups
        self.recommender = recommender
        self.max_int_kmedoids = max_int_kmedoids
        self.parser = parser
        self.user_weights = user_weights

        # internal vars
        self.n_files = 0
        self.train_set_list = []
        self.distance_matrix = None
        self.dir_name = None
        self.gb_train_files = []
        self.weighted_matrices = []
        self.k_users_in_cluster = []

    def read_files(self):
        """
        Method to initialize recommender algorithm.

        """

        self.n_files = len(self.train_files)

        self.users = []
        self.items = []

        for train_file in self.train_files:
            train_set = ReadFile(train_file, sep=self.sep, as_binary=self.as_binary).read()
            self.users += train_set['users']
            self.items += train_set['items']
            self.train_set_list.append(train_set)
            self.dir_name = os.path.dirname(train_file)

        self.users = set(self.users)
        self.items = set(self.items)

        if self.test_file is not None:
            self.test_set = ReadFile(self.test_file).read()
            self.users = sorted(set(list(self.users) + list(self.test_set['users'])))
            self.items = sorted(set(list(self.items) + list(self.test_set['items'])))

        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})

    def compute_distance(self):
        """
        Method to compute a distance matrix from train set

        """

        # Calculate distance matrix
        distance_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))
        # Remove NaNs
        distance_matrix[np.isnan(distance_matrix)] = 1.0

        return distance_matrix

    def create_weighted_matrix(self):
        for ts in self.train_set_list:
            weighted_matrix = np.ones((len(self.users), len(self.users)))

            for u in range(len(self.users)):
                user_u = self.user_id_to_user[u]
                nu = ts['items_seen_by_user'].get(user_u, [])
                if nu:
                    for v in range(u, len(self.users)):
                        if u == v:
                            nuv = len(nu)
                        else:
                            user_v = self.user_id_to_user[v]
                            nv = ts['items_seen_by_user'].get(user_v, [])

                            # nuv = len(set(nu).intersection(nv)) / (len(nu) + len(nv))
                            nuv = 1 / (len(nu) + len(nv))

                        alpha = nuv if nuv != 0 else 1
                        weighted_matrix[u][v] = alpha
                        weighted_matrix[v][u] = alpha
            self.weighted_matrices.append(weighted_matrix)

    def build_distance_matrix(self):
        if self.user_weights:
            self.create_weighted_matrix()

        self.distance_matrix = np.zeros((len(self.users), len(self.users)))

        for n, ts in enumerate(self.train_set_list):
            self.train_set = ts

            self.create_matrix()
            # Missing: Treat distance matrix with feedback
            self.distance_matrix += self.compute_distance()
            if self.user_weights:
                self.distance_matrix /= self.weighted_matrices[n]

        del self.train_set
        # del self.train_set_list

        self.distance_matrix /= self.n_files

    def run_kmedoids(self):
        set_train_tuple = []
        support_matrix, clusters = kmedoids(self.distance_matrix, self.k_groups,
                                            max_interactions=self.max_int_kmedoids, random_seed=123)

        for c, cluster in enumerate(clusters.values()):
            self.k_users_in_cluster.append(len(cluster))
            train_tuple = set()
            for user_id in cluster:
                user = self.user_id_to_user[user_id]
                for tr in self.train_set_list:
                    for item in tr['feedback'].get(user, []):
                        train_tuple.add((user, item, 1))
            train_tuple = sorted(list(train_tuple), key=lambda x: (x[0], x[1]))
            if len(train_tuple) != 0:
                set_train_tuple.append(train_tuple)

        return set_train_tuple

    def generate_groups(self):
        fold_for_sets = self.dir_name + '/gb_train_' + str(self.parser) + '/'
        if not os.path.exists(fold_for_sets):
            os.mkdir(fold_for_sets)

        train_tuple = self.run_kmedoids()
        self.k_groups = len(train_tuple)
        for f in range(len(train_tuple)):
            train_file_name = fold_for_sets + 'train_%d.dat' % f
            WriteFile(train_file_name, data=train_tuple[f], sep=self.sep).write()
            self.gb_train_files.append(train_file_name)
        del self.train_set_list

    def generate_recommendation(self):
        self.ranking = []
        for n, train_file in enumerate(self.gb_train_files):
            if self.recommender == 'UserKNN':
                rec = UserKNN(train_file=train_file, similarity_metric=self.similarity_metric,
                              as_binary=True, as_similar_first=False)
                rec.compute(verbose=False, verbose_evaluation=False)
                self.ranking += rec.ranking

            elif self.recommender == 'ItemKNN':
                rec = ItemKNN(train_file=train_file, test_file=self.test_file,
                              similarity_metric=self.similarity_metric, as_binary=True)
                rec.compute(verbose=False, verbose_evaluation=False)
                self.ranking += rec.ranking

            elif self.recommender == 'MostPopular':
                rec = MostPopular(train_file=train_file, test_file=self.test_file, as_binary=True)
                rec.compute(verbose=False, verbose_evaluation=False)
                self.ranking += rec.ranking

            elif self.recommender == 'BPRMF':
                rec = BprMF(train_file=train_file, test_file=self.test_file, batch_size=4)
                rec.compute(verbose=False, verbose_evaluation=False)
                self.ranking += rec.ranking
            else:
                raise ValueError('Error: Recommender not implemented or not exist!')

        self.ranking = sorted(self.ranking, key=lambda x: (x[0], -x[2]))

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):

        if verbose:
            print("[Case Recommender: Item Recommendation > %s]\n" % self.recommender_name)

        self.read_files()
        self.build_distance_matrix()
        self.generate_groups()
        self.generate_recommendation()

        if verbose:
            print('GroupBased:: Final K value for kmedoids: %d' % self.k_groups)

        self.write_ranking()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)
