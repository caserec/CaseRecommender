# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file contains rating prediction measures:
    MAE
    RMSE
"""

import math
import numpy as np
from framework.utils.extra_functions import check_error_file
from framework.utils.read_file import ReadFile

__author__ = 'Arthur Fortes'


class RatingPredictionEvaluation(object):
    def __init__(self, space_type='\t'):
        self.space_type = space_type

    def simple_evaluation(self, file_result, file_test):
        # Verify that the files are valid
        check_error_file(file_result)
        check_error_file(file_test)

        predict = ReadFile(file_result, space_type=self.space_type)
        predict.main_information()
        test = ReadFile(file_test, space_type=self.space_type)
        test.main_information()

        rmse = 0
        mae = 0
        count_comp = 0
        for user in test.list_users:
            for item in test.dict_users[user]:
                try:
                    rui_predict = float(predict.dict_users[user][item])
                    rui_test = float(test.dict_users[user][item])
                    rmse += math.pow((rui_predict - rui_test), 2)
                    mae += math.fabs(rui_predict - rui_test)
                    count_comp += 1
                except KeyError:
                    pass

        if count_comp != 0:
            rmse = math.sqrt(float(rmse) / float(count_comp))
            mae = math.sqrt(float(mae) / float(count_comp))

        return rmse, mae

    @staticmethod
    def evaluation(predictions, test_set):
        rmse = 0
        mae = 0
        count_comp = 0

        for p in predictions:
            try:
                user, item, rui_predict = p[0], p[1], p[2]
                rui_test = float(test_set["feedback"][user][item])
                rmse += math.pow((rui_predict - rui_test), 2)
                mae += math.fabs(rui_predict - rui_test)
                count_comp += 1
            except KeyError:
                pass
        if count_comp != 0:
            rmse = math.sqrt(float(rmse) / float(count_comp))
            mae = math.sqrt(float(mae) / float(count_comp))

        return rmse, mae

    def folds_evaluation(self, folds_dir, n_folds, name_prediction, name_test):
        list_rmse = list()
        list_mae = list()
        
        for fold in xrange(n_folds):
            prediction = folds_dir + str(fold) + '\\' + name_prediction
            test = folds_dir + str(fold) + '\\' + name_test
            rmse, mae = self.simple_evaluation(prediction, test)
            list_rmse.append(rmse)
            list_mae.append(mae)

        return np.mean(list_rmse), np.std(list_rmse), np.mean(list_mae), np.std(list_mae)
