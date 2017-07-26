# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Item Based Collaborative Filtering Recommender with Attributes

    Its philosophy is as follows: in order to determine the rating of User u on Movie m, we can find other movies that
    are similar to Movie m, and based on User u’s ratings on those similar movies we infer his rating on Movie m.
    However, instead of traditional ItemKNN, this approach uses a pre-computed similarity matrix.

    Literature:
        http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

Parameters
-----------
    - train_file: string
    - test_file: string
    - metadata_file: string
        Metadata file ; Format file:
        item \t metadata \t value\n
    - similarity_matrix_file: file
        Pairwise metric to compute the similarity between the users based on a set of attributes.
        Format file:
        Distances separated by \t, where the users should be ordering. E g.:
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
        distance1\tdistance2\tdistance3\n
    - ranking_file: string
        file to write final ranking
    - neighbors: int
        The number of user candidates strategy that you can choose for selecting the possible items to recommend.
    - similarity_metric: string
        Pairwise metric to compute the similarity between the users.
        Reference about distances:
            - http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    - space_type: string
"""

import sys
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile

__author__ = "Arthur Fortes"


class ItemAttributeKNN(ItemKNN):
    def __init__(self, train_file, test_file=None, metadata_file=None, similarity_matrix_file=None, ranking_file=None,
                 neighbors=30, rank_number=10, similarity_metric="correlation", space_type='\t'):
        ItemKNN.__init__(self, train_file, test_file=test_file, ranking_file=ranking_file, neighbors=neighbors,
                         rank_number=rank_number, similarity_metric=similarity_metric, space_type=space_type)

        if metadata_file is None and similarity_matrix_file is None:
            print("This algorithm needs a similarity matrix or a metadata file!")
            sys.exit(0)

        if metadata_file is not None:
            self.metadata = ReadFile(metadata_file, space_type=space_type).read_metadata(self.items)
            self.matrix = self.metadata['matrix'].T
        self.similarity_matrix_file = similarity_matrix_file

    def read_matrix(self):
        self.si_matrix = ReadFile(self.similarity_matrix_file).read_matrix()

    def execute(self, measures=('Prec@5', 'Prec@10', 'NDCG@5', 'NDCG@10', 'MAP@5', 'MAP@10')):
        print("[Case Recommender: Item Recommendation > Item Attribute KNN Algorithm]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])

        if self.test_file is not None:
            test_set = ReadFile(self.test_file, space_type=self.space_type).return_information()
            print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
                  " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
            del test_set

        if self.similarity_matrix_file is not None:
            print("training time:: ", timed(self.read_matrix), " sec")
        else:
            print("training time:: ", timed(self.compute_similarity), " sec")
            print("prediction_time:: ", timed(self.predict), " sec\n")
        if self.test_file is not None:
            self.evaluate(measures)
