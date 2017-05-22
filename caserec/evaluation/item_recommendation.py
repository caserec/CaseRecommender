# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file contains item recommendation measures:
    MAP
    Precision
    Recall
    
    - default_evaluation: returns:
        * prec@1, recall@1, map@1 ... prec@10, recall@10, map@10 and map_total
        * with only_map=True: map@1, map@3, map@5, map@10 and map_total
        
    - evaluation_ranking: function used for item recommendation algorithms
    
"""

import sys
import numpy as np

from caserec.utils.extra_functions import check_error_file
from caserec.utils.read_file import ReadFile

__author__ = 'Arthur Fortes'


class ItemRecommendationEvaluation(object):
    def __init__(self, space_type='\t', only_map=False, n_ranks=list([1, 3, 5, 10])):
        self.space_type = space_type
        self.only_map = only_map
        self.n_ranks = n_ranks

    def default_evaluation(self, train, test):
        num_user = len(test['users'])
        avg_prec_total = list()
        final_values = list()

        for i, n in enumerate(self.n_ranks):
            if n < 1:
                print('Error: N must >= 1.')
                sys.exit()

            partial_precision = list()
            partial_recall = list()
            avg_prec_total = list()

            for user in test['users']:
                avg_prec_sum = 0

                try:
                    hit_cont = 0
                    # Generate user intersection list between the recommended items and test.
                    list_feedback = set(list(train['du_order'][user])[:n])
                    intersection = list(list_feedback.intersection(test['du_order'][user]))

                    if len(intersection) > 0:
                        partial_precision.append((float(len(intersection)) / float(n)))
                        partial_recall.append((float(len(intersection)) / float(len(test['du'][user]))))

                        for item in intersection:
                            hit_cont += 1
                            avg_prec_sum += (float(hit_cont) / float(list(train['du_order'][user])[:n].index(item) + 1))

                        avg_prec_total.append(float(avg_prec_sum) / float(len(test['du_order'][user])))

                except KeyError:
                    pass

            if not self.only_map:
                final_precision = sum(partial_precision) / float(num_user)
                final_values.append(final_precision)
                final_recall = sum(partial_recall) / float(num_user)
                final_values.append(final_recall)
            final_map = sum(avg_prec_total) / float(num_user)
            final_values.append(final_map)

        final_map_total = sum(avg_prec_total) / float(num_user)
        final_values.append(final_map_total)

        return final_values

    def simple_evaluation(self, file_result, file_test):
        # Verify that the files are valid
        check_error_file(file_result)
        check_error_file(file_test)

        predict = ReadFile(file_result, space_type=self.space_type).return_information()
        test = ReadFile(file_test, space_type=self.space_type).return_information()

        return self.default_evaluation(predict, test)

    def all_but_one_evaluation(self, file_result, file_test):
        check_error_file(file_result)
        check_error_file(file_test)

        predict = ReadFile(file_result, space_type=self.space_type).return_information()
        test = ReadFile(file_test, space_type=self.space_type).return_information()

        for user in test['users']:
            test['du'][user] = [list(test['du'][user])[0]]

        return self.default_evaluation(predict, test)

    def folds_evaluation(self, folds_dir, n_folds, name_prediction, name_test, type_recommendation="SimpleEvaluation",
                         no_deviation=False):
        result = list()
        list_results = list()

        for fold in range(n_folds):
            prediction = folds_dir + str(fold) + '/' + name_prediction
            test = folds_dir + str(fold) + '/' + name_test

            if type_recommendation == "SimpleEvaluation":
                result.append(self.simple_evaluation(prediction, test))
            elif type_recommendation == "AllButOne":
                result.append(self.all_but_one_evaluation(prediction, test))
            else:
                print('Error: Invalid recommendation type!')
                sys.exit()

        for i in range(len(result[0])):
            list_partial = list()
            for j in range(n_folds):
                list_partial.append(result[j][i])
            if no_deviation:
                list_results.append(list_partial)
            else:
                list_results.append([np.mean(list_partial), np.std(list_partial)])

        return list_results

    def evaluation_ranking(self, ranking, test_file):
        ranking_dict = {'du_order': {}}
        test = ReadFile(test_file, space_type=self.space_type).return_information()

        for sample in ranking:
            ranking_dict['du_order'].setdefault(sample[0], list()).append(sample[1])

        return self.default_evaluation(ranking_dict, test)
