import getopt
import sys
import time
from evaluation.rating_prediction import RatingPredictionEvaluation

__author__ = 'Arthur Fortes'

text = '__________________________________________________________________\n' \
       '\n  [Case Recommender Instructions] Evaluation Rating Prediction  \n' \
       '__________________________________________________________________\n\n' \
       'Evaluation Metrics:\n' \
       '\t - Root Mean Square Error (RMSE)\n' \
       '\t - Mean Absolute Error (MAE) \n' \
       '\nCommand: \n'


def main(argv):
    evaluation_type = ''
    predict_file = ''
    test_file = ''
    dir_fold = ''
    num_fold = 10
    space_type = '\t'

    try:
        opts, args = getopt.getopt(argv, "h:e:p:t:d:n:s:",
                                   ["evaluation_type=", "predict_file=", "test_file=", "dir_fold=", "num_fold=",
                                    "space_type="])
    except getopt.GetoptError:
        print(text)

        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(text)
            sys.exit()
        elif opt in ("-s", "--space_type="):
            space_type = arg
            if space_type == 'tabulation':
                space_type = '\t'
            elif space_type == 'comma':
                space_type = ','
            elif space_type == 'dot':
                space_type = '.'
            else:
                print(text)
                sys.exit()
        elif opt in ("-e", "--evaluation_type="):
            evaluation_type = arg
        elif opt in ("-p", "--predict_file="):
            predict_file = arg
        elif opt in ("-t", "--test_file="):
            test_file = arg
        elif opt in ("-d", "--dir_fold="):
            dir_fold = arg
        elif opt in ("-n", "--num_fold="):
            num_fold = arg

    print("\n[Case Recommender - Evaluation Rating Prediction]")
    print "Evaluation type: ", evaluation_type
    print "Predict file: ", predict_file
    print "Test file: ", test_file

    if evaluation_type == "SimpleEvaluation":
        starting_point = time.time()
        a = RatingPredictionEvaluation(space_type)
        rmse, mae = a.simple_evaluation(predict_file, test_file)
        elapsed_time = time.time() - starting_point
        print("Runtime: " + str(elapsed_time / 60) + " second(s) | RMSE: " + str(rmse) + " | MAE: " + str(mae))
        print("\n")

    elif evaluation_type == "FoldsEvaluation":
        print("Folds Path: " + str(dir_fold))
        print("Number of Folds: " + str(num_fold))

        starting_point = time.time()
        a = RatingPredictionEvaluation(space_type)
        rmse, std_rmse, mae, std_mae = a.folds_evaluation(dir_fold, int(num_fold), predict_file, test_file)
        elapsed_time = time.time() - starting_point
        print("Runtime: " + str(elapsed_time / 60) + " second(s)")
        print("\n")
        print("RMSE")
        print("MEAN: " + str(rmse))
        print("STD: " + str(std_rmse))
        print("MAE")
        print("MEAN: " + str(mae))
        print("STD: " + str(std_mae))
        print("\n")

    else:
        print('Error: Invalid evaluation type!')
        print(text)
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
