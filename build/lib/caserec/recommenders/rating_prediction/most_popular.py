# coding=utf-8
""""
    Most Popular Collaborative Filtering Recommender
    [Rating Prediction]

    Most Popular predicts ratings for unobserved items for each user based on popularity of user and items.

"""

# © 2018. Case Recommender (MIT License)

from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from caserec.utils.extra_functions import timed
import numpy as np

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class MostPopular(BaseRatingPrediction):
    def __init__(self, train_file=None, test_file=None, output_file=None, sep='\t', output_sep='\t'):
        """
        Most Popular for Item Recommendation

        This algorithm predicts a rank for each user using the count of number of feedback of users and items

        Usage::

            >> MostPopular(train, test).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        """

        super(MostPopular, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file,
                                          sep=sep, output_sep=output_sep)

        self.recommender_name = 'Most Popular'

    def predict(self):
        """
            This method predict final result, building an rank of each user of the train set.

        """

        if self.test_file is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:

                    count_value = 0
                    feedback_value = 0

                    for user_v in self.train_set['users_viewed_item'].get(item, []):
                        feedback_value += self.train_set['feedback'][user_v][item]
                        count_value += 1

                    if feedback_value == 0:
                        try:
                            feedback_value = np.mean(list(self.train_set['feedback'][user].values()))
                        except KeyError:
                            feedback_value = self.train_set['mean_value']
                    else:
                        feedback_value /= count_value

                    self.predictions.append((user, item, feedback_value))
        else:
            raise NotImplemented

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):
        """
        Extends compute method from BaseItemRecommendation. Method to run recommender algorithm

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

        super(MostPopular, self).compute(verbose=verbose)

        if verbose:
            print("prediction_time:: %4f sec" % timed(self.predict))
            print('\n')

        else:
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)
