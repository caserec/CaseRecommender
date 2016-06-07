"""
    test
"""

from evaluation.rating_prediction import RatingPredictionEvaluation
from recommenders.rating_prediction.item_attribute_knn import ItemAttributeKNN
from recommenders.rating_prediction.itemknn import ItemKNN
from recommenders.rating_prediction.mvlrec import MVLrec
from recommenders.rating_prediction.user_attribute_knn import UserAttributeKNN
from recommenders.rating_prediction.userknn import UserKNN
from utils.read_file import ReadFile
from utils.write_file import WriteFile


class RatingPrediction(object):
    def __init__(self, train_file, recommender, test_file=None, prediction_file=None, similarity_metric="correlation",
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
            print("\n[UserKNN] Number of Neighbors: " + str(neighbors) + " | "
                                                                       "Similarity Metric: " + str(similarity_metric))
            self.predictions = UserKNN(self.train_set, self.test_set, similarity_metric=similarity_metric,
                                       neighbors=neighbors)
        elif self.recommender.lower() == "itemknn":
            print("\n[ItemKNN] Number of Neighbors: " + str(neighbors) + " | "
                                                                       "Similarity Metric: " + str(similarity_metric))
            self.predictions = ItemKNN(self.train_set, self.test_set, similarity_metric=similarity_metric,
                                       neighbors=neighbors)
        elif self.recommender.lower() == "itemattributeknn":
            print("\n[ItemAttributeKNN] Number of Neighbors: " + str(neighbors))
            if distance_matrix is not None:
                self.predictions = ItemAttributeKNN(self.train_set, self.test_set,
                                                    neighbors=neighbors, distance_matrix_file=distance_matrix)
            else:
                print("Error: Invalid Distance Matrix File!")
        elif self.recommender.lower() == "userattributeknn":
            print("\n[UserAttributeKNN] Number of Neighbors: " + str(neighbors))
            if distance_matrix is not None:
                self.predictions = UserAttributeKNN(self.train_set, self.test_set, neighbors=neighbors,
                                                    distance_matrix_file=distance_matrix)
            else:
                print("Error: Invalid Distance Matrix File!")
        elif self.recommender.lower() == "mvlrec":
            print("\n[MVLrec]")
            self.predictions = MVLrec(self.train_set, percent=0.2, recommender1="itemknn", recommender2="userknn",
                                      times=20, k=1000)
        else:
            print("Error: Invalid Recommender!")

        if self.predictions:
            if prediction_file is not None:
                WriteFile(prediction_file, self.predictions.predictions, space_type)
            rmse, mae = RatingPredictionEvaluation().evaluation(self.predictions.predictions, self.test_set)
            print("RMSE: " + str(rmse) + " MAE: " + str(mae) + "\n")
        else:
            print("Error: No predictions!")

RatingPrediction("C:/Users/Arthur/OneDrive/ml100k/folds/0/train.dat",
                 test_file="C:/Users/Arthur/OneDrive/ml100k/folds/0/test.dat", recommender="userknn")

#
# RatingPrediction("C:/Users/Arthur/OneDrive/ml100k/folds/0/train.dat",
#                  test_file="C:/Users/Arthur/OneDrive/ml100k/folds/0/test.dat", recommender="itemknn")
