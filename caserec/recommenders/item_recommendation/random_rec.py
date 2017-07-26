# coding=utf-8
"""
© 2017. Case Recommender All Rights Reserved (License GPL3)

Random Recommender


Parameters
-----------
    - train_file: string
    - test_file: string
    - ranking_file: string
        file to write final ranking
    - rank_number int
        The number of items per user that appear in final rank
    - space_type: string

"""
import random

from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from caserec.utils.extra_functions import timed
from caserec.utils.read_file import ReadFile
from caserec.utils.write_file import WriteFile


class RandomRec(object):
    def __init__(self, train_file, test_file=None, ranking_file=None, rank_number=10, space_type='\t'):
        self.train_set = ReadFile(train_file, space_type=space_type).return_information()
        self.test_file = test_file
        self.users = self.train_set['users']
        self.items = self.train_set['items']
        if self.test_file is not None:
            self.test_set = ReadFile(test_file).return_information()
            self.users = sorted(list(self.train_set['users']) + list(self.test_set['users']))
            self.items = sorted(list(self.train_set['items']) + list(self.test_set['items']))
        self.ranking_file = ranking_file
        self.rank_number = rank_number
        self.space_type = space_type
        self.ranking = list()

    def predict(self):
        for user in set(self.users):
            rank_user = list()
            for item in self.train_set['not_seen'].get(user, []):
                rank_user.append((user, item, random.uniform(0, 1)))
            rank_user = sorted(rank_user, key=lambda x: -x[2])
            self.ranking += rank_user[:self.rank_number]

        if self.ranking_file is not None:
            WriteFile(self.ranking_file, self.ranking).write_recommendation()

    def evaluate(self, measures):
        res = ItemRecommendationEvaluation().evaluation_ranking(self.ranking, self.test_file)
        evaluation = 'Eval:: '
        for measure in measures:
            evaluation += measure + ': ' + str(res[measure]) + ' '
        print(evaluation)

    def execute(self, measures=('Prec@5', 'Prec@10', 'NDCG@5', 'NDCG@10', 'MAP@5', 'MAP@10')):
        print("[Case Recommender: Item Recommendation > Random Algorithm]\n")
        print("training data:: ", len(self.train_set['users']), " users and ", len(self.train_set['items']),
              " items and ", self.train_set['ni'], " interactions | sparsity ", self.train_set['sparsity'])

        if self.test_file is not None:
            test_set = ReadFile(self.test_file, space_type=self.space_type).return_information()
            print("test data:: ", len(self.test_set['users']), " users and ", len(self.test_set['items']),
                  " items and ", (self.test_set['ni']), " interactions | sparsity ", self.test_set['sparsity'])
            del test_set
        print("prediction_time:: ", timed(self.predict), " sec\n")
        if self.test_file is not None:
            self.evaluate(measures)
