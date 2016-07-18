import time

from evaluation.rating_prediction import RatingPredictionEvaluation
from utils.read_file import ReadFile
from utils.write_file import WriteFile

from framework.recommenders import ItemAttributeKNN
from framework.recommenders import ItemKNN
from framework.recommenders import MatrixFactorization
from framework.recommenders import UserAttributeKNN
from framework.recommenders import UserKNN

__author__ = 'Arthur Fortes'

"""
    Recommenders:
    - ItemKNN
    - ItemAttributeKNN
    - UserKNN
    - UserAttributeKNN
    - MatrixFactorization
"""


class RatingPrediction(object):
    def __init__(self, train_file, test_file, recommender, prediction_file=None, similarity_metric="correlation",
                 neighbors=30, distance_matrix=None, space_type="\t", steps=30, gamma=0.01, delta=0.015, factors=10,
                 init_mean=0.1, init_stdev=0.1, baseline=False):

        self.recommender = str(recommender)

        # print information about dataset
        print("[Case Recommender - Rating Prediction]")
        self.train_set = ReadFile(train_file).rating_prediction()
        print("training data: " + str(len(self.train_set["users"])) + " users " + str(len(self.train_set["items"])) +
              " items " + str(self.train_set["ni"]) + " ratings")
        self.test_set = ReadFile(test_file).rating_prediction()
        print("test data: " + str(len(self.test_set["users"])) + " users " + str(len(self.test_set["items"])) +
              " items " + str(self.test_set["ni"]) + " ratings")

        # run recommenders

        if self.recommender.lower() == "userknn":
            rec = UserKNN(self.train_set, self.test_set, similarity_metric=similarity_metric, neighbors=neighbors)
            starting_point = time.time()
            rec.compute_similarity()
            elapsed_time = time.time() - starting_point
            print("- Training time: " + str(elapsed_time) + " second(s)")
            starting_point = time.time()
            self.predictions = rec.predict()
            elapsed_time = time.time() - starting_point
            print("- Prediction time: " + str(elapsed_time) + " second(s)")

        elif self.recommender.lower() == "itemknn":
            rec = ItemKNN(self.train_set, self.test_set, similarity_metric=similarity_metric, neighbors=neighbors)
            starting_point = time.time()
            rec.compute_similarity()
            elapsed_time = time.time() - starting_point
            print("- Training time: " + str(elapsed_time) + " second(s)")
            starting_point = time.time()
            self.predictions = rec.predict()
            elapsed_time = time.time() - starting_point
            print("- Prediction time: " + str(elapsed_time) + " second(s)")

        elif self.recommender.lower() == "itemattributeknn":
            if distance_matrix is not None:
                starting_point = time.time()
                rec = ItemAttributeKNN(self.train_set, self.test_set, neighbors=neighbors,
                                       distance_matrix_file=distance_matrix)
                rec.read_matrix()
                elapsed_time = time.time() - starting_point
                print("- Training time: " + str(elapsed_time) + " second(s)")
                starting_point = time.time()
                self.predictions = rec.predict()
                elapsed_time = time.time() - starting_point
                print("- Prediction time: " + str(elapsed_time) + " second(s)")
            else:
                print("Error: Invalid Distance Matrix File!")

        elif self.recommender.lower() == "userattributeknn":
            if distance_matrix is not None:
                starting_point = time.time()
                rec = UserAttributeKNN(self.train_set, self.test_set, neighbors=neighbors,
                                       distance_matrix_file=distance_matrix)
                rec.read_matrix()
                elapsed_time = time.time() - starting_point
                print("- Training time: " + str(elapsed_time) + " second(s)")
                starting_point = time.time()
                self.predictions = rec.predict()
                elapsed_time = time.time() - starting_point
                print("- Prediction time: " + str(elapsed_time) + " second(s)")
            else:
                print("Error: Invalid Distance Matrix File!")

        elif self.recommender.lower() == "matrixfactorization":
            rec = MatrixFactorization(self.train_set, self.test_set, steps, gamma, delta, factors, init_mean,
                                      init_stdev, baseline)
            starting_point = time.time()
            rec.train_mf()
            elapsed_time = time.time() - starting_point
            print("- Training time: " + str(elapsed_time) + " second(s)")
            starting_point = time.time()
            self.predictions = rec.predict()
            elapsed_time = time.time() - starting_point
            print("- Prediction time: " + str(elapsed_time) + " second(s)")
        else:
            print("Error: Invalid Recommender!")

        # Evaluation and Write Predictions

        if self.predictions:
            # for single recommenders
            if type(self.predictions) is list:
                if prediction_file is not None:
                    WriteFile(prediction_file, self.predictions, space_type).write_prediction_file()
                rmse, mae = RatingPredictionEvaluation().evaluation(self.predictions, self.test_set)
                print("RMSE: " + str(rmse) + " MAE: " + str(mae) + "\n")

            # for multiple recommenders or ensembles
            elif type(self.predictions) is dict:
                print("\n")
                for d in self.predictions:
                    print("[" + d + "]")
                    for p in self.predictions[d]:
                        rmse, mae = RatingPredictionEvaluation().evaluation(p[1], self.test_set)
                        print("[" + p[0] + "] RMSE: " + str(rmse) + " MAE: " + str(mae))
        else:
            print("Error: No predictions!")

RatingPrediction("C:/Users/Arthur/OneDrive/ml100k/folds/0/train.dat",
                 "C:/Users/Arthur/OneDrive/ml100k/folds/0/test.dat",
                 recommender="matrixfactorization")
