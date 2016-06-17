import sys
import numpy as np
from utils.error_functions import check_error_file
from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'


class ItemRecommendationEvaluation(object):
    def __init__(self, space_type='\t', only_map=False):
        self.space_type = space_type
        self.only_map = only_map

    def simple_evaluation(self, file_result, file_test, n_ranks=list([1, 3, 5, 10])):
        # Verify that the files are valid
        check_error_file(file_result)
        check_error_file(file_test)

        predict = ReadFile(file_result, space_type=self.space_type)
        predict.main_information_item_recommendation()
        test = ReadFile(file_test, space_type=self.space_type)
        test.main_information_item_recommendation()

        num_user = len(test.list_users)
        avg_prec_total = list()
        final_values = list()

        for i, n in enumerate(n_ranks):
            if n < 1:
                print('Error: N must >= 1.')
                sys.exit()

            partial_precision = list()
            partial_recall = list()
            avg_prec_total = list()

            for user in test.list_users:
                hit_cont = 0
                avg_prec_sum = 0

                try:
                    # Generate user intersection list between the recommended items and test.
                    intersection = list(set(predict.dict_users[user][:n]).intersection(
                        set(test.dict_users[user])))

                    if len(intersection) > 0:
                        partial_precision.append((float(len(intersection)) / float(n)))
                        partial_recall.append((float(len(intersection)) / float(test.num_user_interactions[user])))

                        for item in intersection:
                            hit_cont += 1
                            avg_prec_sum += (float(hit_cont) / float(test.dict_users[user].index(item) + 1))

                        avg_prec_total.append(float(avg_prec_sum) / float(test.num_user_interactions[user]))

                except KeyError:
                    pass

            final_precision = sum(partial_precision) / float(num_user)
            final_values.append(final_precision)
            final_recall = sum(partial_recall) / float(num_user)
            final_values.append(final_recall)

        final_map_total = sum(avg_prec_total) / float(num_user)
        final_values.append(final_map_total)

        return final_values

    def all_but_one_evaluation(self, file_result, file_test, n_ranks=list([1, 3, 5, 10])):
        check_error_file(file_result)
        check_error_file(file_test)

        predict = ReadFile(file_result, space_type=self.space_type)
        predict.main_information_item_recommendation()
        test = ReadFile(file_test, space_type=self.space_type)
        test.main_information_item_recommendation()

        num_user = len(test.list_users)
        final_values = list()

        for user in test.list_users:
            test.dict_users[user] = [test.dict_users[user][0]]

        for i, n in enumerate(n_ranks):
            if n < 1:
                print('Error: N must >= 1.')
                sys.exit()

            partial_precision = list()
            partial_recall = list()
            avg_prec_total = list()

            for user in test.list_users:
                num_user_interactions = len(test.dict_users[user])
                hit_cont = 0
                avg_prec_sum = 0

                try:
                    # Generate user intersection list between the recommended items and test.
                    intersection = list(set(predict.dict_users[user][:n]).intersection(
                        set(test.dict_users[user])))

                    if len(intersection) > 0:
                        partial_precision.append((float(len(intersection)) / float(n)))
                        partial_recall.append((float(len(intersection)) / float(num_user_interactions)))

                        for item in intersection:
                            hit_cont += 1
                            avg_prec_sum += (float(hit_cont) / float(test.dict_users[user].index(item) + 1))

                        avg_prec_total.append(float(avg_prec_sum) / float(num_user_interactions))

                except KeyError:
                    pass

            if not self.only_map:
                final_precision = sum(partial_precision) / float(num_user)
                final_values.append(final_precision)
                final_recall = sum(partial_recall) / float(num_user)
                final_values.append(final_recall)
            final_map = sum(avg_prec_total) / float(num_user)
            final_values.append(final_map)

        return final_values

    def folds_evaluation(self, folds_dir, n_folds, name_prediction, name_test, type_recommendation="SimpleEvaluation",
                         n_ranks=list([1, 3, 5, 10]), no_deviation=False):
        result = list()
        list_results = list()

        for fold in xrange(n_folds):
            prediction = folds_dir + str(fold) + '\\' + name_prediction
            test = folds_dir + str(fold) + '\\' + name_test

            if type_recommendation == "SimpleEvaluation":
                result.append(self.simple_evaluation(prediction, test, n_ranks))
            elif type_recommendation == "AllButOne":
                result.append(self.all_but_one_evaluation(prediction, test, n_ranks))
            else:
                print('Error: Invalid recommendation type!')
                sys.exit()

        for i in xrange(len(result[0])):
            list_partial = list()
            for j in xrange(n_folds):
                list_partial.append(result[j][i])
            if no_deviation:
                list_results.append(list_partial)
            else:
                list_results.append([np.mean(list_partial), np.std(list_partial)])

        return list_results
