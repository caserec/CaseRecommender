# coding=utf-8
""""
    Random Collaborative Filtering Recommender
    [Rating Prediction (Rating)]

    Random predicts a user’s ratings based on random distributions of rates.

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from caserec.utils.extra_functions import timed

__author__ = 'Fernando S. de Aguiar Neto <fsan110792@gmail.com>'


class RandomRec(BaseRatingPrediction):
    def __init__(self, train_file, test_file, uniform=True, output_file=None, sep='\t', output_sep='\t',
                 random_seed=None):
        """
        Random recommendation for Rating Prediction

        This algorithm predicts ratings for each user-item

        Usage::

            >> RandomRec(train, test).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None
        
        :param uniform: Indicates whether the ratings are drawn from a uniform sample or not
        if False, the ratings are drawn from a normal distribution with the same mean and standard deviation
        as the feedback provided in train
        :type uniform: bool, default True

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None
        
        """

        super(RandomRec, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                        sep=sep, output_sep=output_sep)

        if random_seed is not None:
            np.random.seed(random_seed)

        self.uniform = uniform

        self.recommender_name = 'Random Recommender'

    def predict(self):
        if not self.uniform:
            feedbacks = []
            for user in self.train_set["users"]:
                for item in self.train_set['items_seen_by_user'][user]:
                    feedbacks.append(self.train_set['feedback'][user][item])

            std = np.std(feedbacks)

        if self.test_file is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    if self.uniform:
                        feedback_value = np.random.uniform(self.train_set['min_value'], self.train_set['max_value'])
                    else:
                        feedback_value = np.random.normal(self.train_set['mean_value'], std)

                    self.predictions.append((user, item, feedback_value))
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

        super(RandomRec, self).compute(verbose=verbose)

        if verbose:
            print("prediction_time:: %4f sec" % timed(self.predict))
            print('\n')

        else:
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)
