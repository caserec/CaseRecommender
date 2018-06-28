# coding=utf-8
"""
    UserKNN based on Collaborative Filtering Recommender
    [Rating Prediction]

    Literature:
        KAggarwal, Charu C.:
        Chapter 2: Neighborhood-Based Collaborative Filtering
        Recommender Systems: The Textbook. 2016
        file:///home/fortesarthur/Documentos/9783319296579-c1.pdf

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

from caserec.utils.extra_functions import timed
from caserec.recommenders.rating_prediction.base_knn import BaseKNN

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class UserKNN(BaseKNN):
    def __init__(self, train_file=None, test_file=None, output_file=None, similarity_metric="cosine", k_neighbors=None,
                 as_similar_first=False, sep='\t', output_sep='\t'):
        """
        UserKNN for rating prediction

        This algorithm predicts ratings for each user based on the similar items that his neighbors
        (similar users) consumed.

        Usage::

            >> UserKNN(train, test).compute()
            >> UserKNN(train, test, ranking_file, as_similar_first=True, k_neighbors=60).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param similarity_metric: Pairwise metric to compute the similarity between the users. Reference about
        distances: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
        :type similarity_metric: str, default cosine

        :param k_neighbors: Number of neighbors to use. If None, k_neighbor = int(sqrt(n_users))
        :type k_neighbors: int, default None

        :param as_similar_first: If True, for each unknown item, which will be predicted, we first look for its k
        most similar users and then take the intersection with the users that
        seen that item.
        :type as_similar_first: bool, default False

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        """
        super(UserKNN, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                      similarity_metric=similarity_metric, sep=sep, output_sep=output_sep)

        self.recommender_name = 'UserKNN Algorithm'

        self.as_similar_first = as_similar_first
        self.k_neighbors = k_neighbors

        # internal vars
        self.su_matrix = None
        self.users_id_viewed_item = None

    def init_model(self):
        """
        Method to initialize the model. Compute similarity matrix based on user (user x user)

        """

        super(UserKNN, self).init_model()

        self.users_id_viewed_item = {}

        # Set the value for k
        if self.k_neighbors is None:
            self.k_neighbors = int(np.sqrt(len(self.users)))

        self.su_matrix = self.compute_similarity(transpose=False)

        # Map the users which seen an item with their respective ids
        for item in self.items:
            for user in self.train_set['users_viewed_item'].get(item, []):
                self.users_id_viewed_item.setdefault(item, []).append(self.user_to_user_id[user])

    def predict(self):
        """
        Method to predict ratings for all known users in the train set.

        """

        for user in self.users:
            if len(self.train_set['feedback'].get(user, [])) != 0:
                if self.test_file is not None:
                    if self.as_similar_first:
                        self.predictions += self.predict_similar_first_scores(user, self.test_set['items_seen_by_user']
                                                                              .get(user, []))
                    else:
                        self.predictions += self.predict_scores(user, self.test_set['items_seen_by_user'].get(user, []))
                else:
                    # Selects items that user has not interacted with.
                    items_seen_by_user = []
                    u_list = list(np.flatnonzero(self.matrix[self.user_to_user_id[user]] == 0))
                    for item_id in u_list:
                        items_seen_by_user.append(self.item_id_to_item[item_id])

                    if self.as_similar_first:
                        self.predictions += self.predict_similar_first_scores(user, items_seen_by_user)
                    else:
                        self.predictions += self.predict_scores(user, items_seen_by_user)
            else:
                # Implement cold start user
                pass

    def predict_scores(self, user, unpredicted_items):
        """
        In this implementation, for each unknown item,
        which will be predicted, we first look for users that seen that item and calculate the similarity between them
        and the user. Then we sort these similarities and get the most similar k's. Finally, the score of the
        unknown item will be the sum of the similarities.

        rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

        :param user: User
        :type user: int

        :param unpredicted_items: A list of unknown items for each user
        :type unpredicted_items: list

        :return: Sorted list with triples user item rating
        :rtype: list

        """

        u_id = self.user_to_user_id[user]
        predictions = []

        for item in unpredicted_items:
            neighbors = []
            rui = 0
            sim_sum = 0
            for user_v_id in self.users_id_viewed_item.get(item, []):
                user_v = self.user_id_to_user[user_v_id]
                neighbors.append((user_v, self.su_matrix[u_id, user_v_id], self.train_set['feedback'][user_v][item]))
            neighbors = sorted(neighbors, key=lambda x: -x[1])

            if neighbors:
                for triple in neighbors[:self.k_neighbors]:
                    rui += (triple[2] - self.bui[triple[0]][item]) * triple[1] if triple[1] != 0 else 0.001
                    sim_sum += triple[1] if triple[1] != 0 else 0.001

                rui = self.bui[user][item] + (rui / sim_sum)

            else:
                rui = self.bui[user][item]

            # normalize the ratings based on the highest and lowest value.
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

            predictions.append((user, item, rui))

        return sorted(predictions, key=lambda x: x[1])

    def predict_similar_first_scores(self, user, unpredicted_items):
        """
        In this implementation, for each unknown item, which will be
        predicted, we first look for its k most similar users and then take the intersection with the users that
        seen that item. Finally, the score of the unknown item will be the sum of the  similarities.

        rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

        :param user: User
        :type user: int

        :param unpredicted_items: A list of unknown items for each user
        :type unpredicted_items: list

        :return: Sorted list with triples user item rating
        :rtype: list

        """
        u_id = self.user_to_user_id[user]
        predictions = []

        # Select user neighbors, sorting user similarity vector. Returns a list with index of sorting values
        neighbors = sorted(range(len(self.su_matrix[u_id])), key=lambda m: -self.su_matrix[u_id][m])

        for item in unpredicted_items:
            rui = 0
            sim_sum = 0

            # Intersection bt. the neighbors closest to the user and the users who accessed the unknown item.
            common_users = list(set(
                self.users_id_viewed_item.get(item, [])).intersection(neighbors[1:self.k_neighbors]))

            if common_users:
                for user_v_id in common_users:
                    user_v = self.user_id_to_user[user_v_id]
                    sim_uv = self.su_matrix[u_id, user_v_id]
                    rui += (self.train_set['feedback'][user_v][item] - self.bui[user_v][item]) * \
                        sim_uv if sim_sum != 0 else 0.001
                    sim_sum += sim_uv if sim_sum != 0 else 0.001

                rui = self.bui[user][item] + (rui / sim_sum)

            else:
                rui = self.bui[user][item]

            # normalize the ratings based on the highest and lowest value.
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

            predictions.append((user, item, rui))

        return sorted(predictions, key=lambda x: x[1])

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):
        """
        Extends compute method from BaseItemRecommendation. Method to run recommender algorithm

        :param verbose: Print recommender and database information
        :type verbose: bool, default True

        :param metrics: List of evaluation metrics
        :type metrics: list, default None

        :param verbose_evaluation: Print the evaluation results
        :type verbose_evaluation: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        super(UserKNN, self).compute(verbose=verbose)

        if verbose:
            self.init_model()
            print("training_time:: %4f sec" % timed(self.train_baselines))
            if self.extra_info_header is not None:
                print(self.extra_info_header)
            print("prediction_time:: %4f sec" % timed(self.predict))

        else:
            # Execute all in silence without prints
            self.extra_info_header = None
            self.init_model()
            self.train_baselines()
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)
