# coding=utf-8
import numpy as np
from scipy.spatial.distance import squareform, pdist
from CaseRecommender.utils.read_file import ReadFile
from CaseRecommender.recommenders.rating_prediction.base_knn import BaseKNNRecommenders

__author__ = 'Arthur Fortes'

"""

User Based Collaborative Filtering Recommender

User-kNN predicts a user’s rating according to how similar users rated the same item. The algorithm matches similar
users based on the similarity of their ratings on items.

More details: http://files.grouplens.org/papers/algs.pdf

Parameters
-----------
    - similarity_metric: string
        Pairwise metric to compute the similarity between the users.
        Reference about distances:
            - http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    - neighbors: int
        The number of user candidates strategy that you can choose for selecting the possible items to recommend.

"""


class UserKNN(BaseKNNRecommenders):
    def __init__(self, train_file, test_file, similarity_metric="correlation", neighbors=30):
        train_set = ReadFile(train_file).rating_prediction()
        test_set = ReadFile(test_file).rating_prediction()
        BaseKNNRecommenders.__init__(self, train_set, test_set)
        self.k = neighbors
        self.similarity_metric = similarity_metric
        self.predictions = list()
        self.su_matrix = None

        # methods
        self.train_baselines()

    def compute_similarity(self):
        # Calculate distance matrix between users
        self.su_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))
        # transform distances in similarities
        self.su_matrix = 1 - self.su_matrix
        del self.matrix

    '''
    for each pair (u,i) in test set, this method returns a prediction based
    on the others feedback in the train set

    rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

    '''
    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item in self.test['feedback'][user]:
                    list_n = list()
                    try:
                        ruj = 0
                        sum_sim = 0

                        for user_j in self.train['di'][item]:
                            sim = self.su_matrix[self.map_users[user]][self.map_users[user_j]]
                            list_n.append((user_j, sim))
                        list_n = sorted(list_n, key=lambda x: -x[1])

                        for pair in list_n[:self.k]:
                            ruj += (self.train['feedback'][pair[0]][item] - self.bui[pair[0]][item]) * pair[1]
                            sum_sim += pair[1]

                        ruj = self.bui[user][item] + (ruj / sum_sim)

                        # normalize the ratings based on the highest and lowest value.
                        if ruj > 5:
                            ruj = 5.0
                        if ruj < 0.5:
                            ruj = 0.5
                        self.predictions.append((user, item, ruj))

                    except KeyError:
                        pass
            return self.predictions
