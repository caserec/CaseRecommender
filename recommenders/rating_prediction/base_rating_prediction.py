"""
    test
"""
from recommenders.rating_prediction.item_attribute_knn import ItemAttributeKNN
from recommenders.rating_prediction.itemknn import ItemKNN
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
        if test_file is not None:
            self.test_set = ReadFile(test_file).rating_prediction()
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
                self.predictions = ItemAttributeKNN(self.train_set, self.test_set, similarity_metric=similarity_metric,
                                                    neighbors=neighbors, distance_matrix_file=distance_matrix)
            else:
                print("Error: Invalid Distance Matrix!")
        elif self.recommender.lower() == "userattributeknn":
            if distance_matrix is not None:
                self.predictions = UserAttributeKNN(self.train_set, self.test_set, similarity_metric=similarity_metric,
                                                    neighbors=neighbors, distance_matrix_file=distance_matrix)
            else:
                print("Error: Invalid Distance Matrix!")
        else:
            print("Error: Invalid Recommender!")

        if self.predictions:
            WriteFile(prediction_file, self.predictions, space_type)
        else:
            print("Error: No predictions!")
