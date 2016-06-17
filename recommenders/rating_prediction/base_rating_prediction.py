"""
    test
"""

from evaluation.rating_prediction import RatingPredictionEvaluation
from recommenders.rating_prediction.item_attribute_knn import ItemAttributeKNN
from recommenders.rating_prediction.itemknn import ItemKNN
from recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from recommenders.rating_prediction.mvlrec import MVLrec
from recommenders.rating_prediction.user_attribute_knn import UserAttributeKNN
from recommenders.rating_prediction.userknn import UserKNN
from utils.read_file import ReadFile
from utils.write_file import WriteFile


class RatingPrediction(object):
    def __init__(self, train_file, test_file, recommender, prediction_file=None, similarity_metric="correlation",
                 neighbors=30, distance_matrix=None, space_type="\t"):
        self.recommender = str(recommender)
        self.predictions = list()
        self.train_set = ReadFile(train_file).rating_prediction()

        print("[Case Recommender - Rating Prediction]")
        print("training data: " + str(len(self.train_set["users"])) + " users " + str(len(self.train_set["items"])) +
              " items " + str(self.train_set["ni"]) + " ratings")
        if test_file is not None:
            self.test_set = ReadFile(test_file).rating_prediction()
            print("test data: " + str(len(self.test_set["users"])) + " users " + str(len(self.test_set["items"])) +
                  " items " + str(self.test_set["ni"]) + " ratings")
        else:
            self.test_set = None

        if self.recommender.lower() == "userknn":
            self.predictions = UserKNN(self.train_set, self.test_set, similarity_metric=similarity_metric,
                                       neighbors=neighbors)
        elif self.recommender.lower() == "itemknn":
            self.predictions = ItemKNN(self.train_set, self.test_set, similarity_metric=similarity_metric,
                                       neighbors=neighbors)
        elif self.recommender.lower() == "itemattributeknn":
            if distance_matrix is not None:
                self.predictions = ItemAttributeKNN(self.train_set, self.test_set,
                                                    neighbors=neighbors, distance_matrix_file=distance_matrix)
            else:
                print("Error: Invalid Distance Matrix File!")
        elif self.recommender.lower() == "userattributeknn":
            if distance_matrix is not None:
                self.predictions = UserAttributeKNN(self.train_set, self.test_set, neighbors=neighbors,
                                                    distance_matrix_file=distance_matrix)
            else:
                print("Error: Invalid Distance Matrix File!")
        elif self.recommender.lower() == "mvlrec":
            print("\n[MVLrec]")
            self.predictions = MVLrec(self.train_set, self.test_set, percent=0.8, recommender1="itemknn",
                                      recommender2="userknn", times=19, k=1000)
        elif self.recommender.lower() == "matrixfactorization":
            self.predictions = MatrixFactorization(self.train_set, self.test_set)
        else:
            print("Error: Invalid Recommender!")

        if self.predictions.predictions:
            if type(self.predictions.predictions) is list:
                if prediction_file is not None:
                    WriteFile(prediction_file, self.predictions.predictions, space_type).write_prediction_file()
                rmse, mae = RatingPredictionEvaluation().evaluation(self.predictions.predictions, self.test_set)
                print("RMSE: " + str(rmse) + " MAE: " + str(mae) + "\n")
            elif type(self.predictions.predictions) is dict:
                print("\n")
                for d in self.predictions.predictions:
                    print("[" + d + "]")
                    for p in self.predictions.predictions[d]:
                        rmse, mae = RatingPredictionEvaluation().evaluation(p[1], self.test_set)
                        print("[" + p[0] + "] RMSE: " + str(rmse) + " MAE: " + str(mae))
        else:
            print("Error: No predictions!")

RatingPrediction("C:/Users/Arthur/OneDrive/ml100k/folds/0/train.dat",
                 "C:/Users/Arthur/OneDrive/ml100k/folds/0/test.dat", recommender="matrixfactorization",
                 prediction_file="C:/Users/Arthur/OneDrive/ml100k/folds/0/p.dat")
