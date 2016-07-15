# coding=utf-8
from scipy.spatial.distance import squareform, pdist
from recommenders.rating_prediction.base_knn import BaseKNNRecommenders
import numpy as np

from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'

"""
Item Based Collaborative Filtering Recommender

Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that are
similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.

More details: http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf


Parameters
-----------
    - similarity_metric: string
        Pairwise metric to compute the similarity between the users.
        Reference about distances:
            - http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    - neighbors: int
        The number of item candidates strategy that you can choose for selecting the possible items to recommend.

"""


class ItemKNN(BaseKNNRecommenders):
    def __init__(self, train_file, test_file, similarity_metric="correlation", neighbors=30):
        train_set = ReadFile(train_file).rating_prediction()
        test_set = ReadFile(test_file).rating_prediction()
        BaseKNNRecommenders.__init__(self, train_set, test_set)
        self.k = neighbors
        self.similarity_metric = similarity_metric
        self.predictions = list()
        self.si_matrix = None

        # methods
        # training baselines bui
        self.train_baselines()

    def compute_similarity(self):
        # Calculate distance matrix between items
        self.si_matrix = np.float32(squareform(pdist(self.matrix.T, self.similarity_metric)))
        # transform distances in similarities
        self.si_matrix = 1 - self.si_matrix
        del self.matrix

    '''
     for each pair (u,i) in test set, this method returns a prediction based
     on the others feedback in the train set

     rui = bui + (sum((ruj - buj) * sim(i,j)) / sum(sim(i,j)))

    '''
    def predict(self):
        if self.test is not None:
            for user in self.test['users']:
                for item_j in self.test['feedback'][user]:
                    list_n = list()
                    try:
                        ruj = 0
                        sum_sim = 0
                        for item_i in self.train['feedback'][user]:
                            try:
                                sim = self.si_matrix[self.map_items[item_i]][self.map_items[item_j]]
                            except KeyError:
                                sim = 0
                            list_n.append((item_i, sim))
                        list_n = sorted(list_n, key=lambda x: -x[1])

                        for pair in list_n[:self.k]:
                            ruj += (self.train['feedback'][user][pair[0]] - self.bui[user][pair[0]]) * pair[1]
                            sum_sim += pair[1]
                        ruj = self.bui[user][item_j] + (ruj / sum_sim)

                        # normalize the ratings based on the highest and lowest value.
                        if ruj > 5:
                            ruj = 5.0
                        if ruj < 0.5:
                            ruj = 0.5

                        self.predictions.append((user, item_j, ruj))

                    except KeyError:
                        pass
            return self.predictions
