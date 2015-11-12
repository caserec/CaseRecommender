import sys
import time
from evaluation.rating_prediction import RatingPredictionEvaluation

__author__ = 'Arthur Fortes'

text = '________________________________________________________________\n' \
       '\n  [Case Recommender] Rating Prediction Evaluation Instructions  \n' \
       '________________________________________________________________\n\n' \
       'Evaluation Metrics:\n' \
       '\t - Root Mean Square Error (RMSE)\n' \
       '\t - Mean Absolute Error (MAE) \n' \
       '\nCommand: \n' \
       '\tpython evaluation_rating_prediction.py type_prediction prediction_file ... \n' \
       '\n' \
       '* type_prediction:\n' \
       '\t - SimpleEvaluation: >> python evaluation_rating_prediction.py SimpleEvaluation ' \
       'prediction_file test_file\n\n' \
       '\t - FoldsEvaluation: >> python evaluation_rating_prediction.py FoldsEvaluation ' \
       'folds_directory number_of_folds prediction_file_name test_file_name\n\n' \
       '\t\t.folds_directory: only the directory of the folder contents. \n' \
       '\t\t.number_of_folds: number of folds from cross fold validation\n' \
       '\t\t.prediction_file_name and test_file_name: put only the name and extension: file.dat\n' \
       '\n' \
       '* Optional: you can add space type from your files.\n' \
       'Types:\n' \
       '\t - tabulation\n' \
       '\t - comma\n' \
       '\t - dot\n' \
       'e.g: >> python evaluation_rating_prediction.py SimpleEvaluation prediction_file.dat test.dat tabulation\n'


def main(argv):
    try:
        if argv[0] == 'SimpleEvaluation':
            st = '\t'
            try:
                try:
                    if str(argv[3]):
                        if str(argv[3]) == 'tabulation':
                            st = '\t'
                        elif str(argv[3]) == 'comma':
                            st = ','
                        elif str(argv[3]) == 'dot':
                            st = '.'
                        else:
                            print('Erro: Invalid space type!')
                            print(text)
                            sys.exit()
                except IndexError:
                    pass

                starting_point = time.time()
                a = RatingPredictionEvaluation(space_type=str(st))
                rmse, mae = a.simple_evaluation(str(argv[1]), str(argv[2]))
                elapsed_time = time.time() - starting_point

                print("Runtime: " + str(elapsed_time / 60) + " second(s) | RMSE: " + str(rmse) + " | MAE: " + str(mae))

            except IndexError:
                print(text)
                sys.exit()

        if argv[0] == 'FoldsEvaluation':
            st = '\t'
            try:
                try:
                    if str(argv[5]):
                        if str(argv[5]) == 'tabulation':
                            st = '\t'
                        elif str(argv[5]) == 'comma':
                            st = ','
                        elif str(argv[5]) == 'dot':
                            st = '.'
                        else:
                            print(text)
                            sys.exit()
                except IndexError:
                    pass

                starting_point = time.time()
                a = RatingPredictionEvaluation(space_type=str(st))
                rmse, std_rmse, mae, std_mae = a.folds_evaluation(str(argv[1]), int(argv[2]), str(argv[3]), str(argv[4]))
                elapsed_time = time.time() - starting_point

                print("\nNumber of folds: " + str(argv[2]) + " | Runtime: " + str(elapsed_time / 60) +
                      " second(s) \nMean (RMSE): " + str(rmse) + " "
                      "| Standard Deviation (RMSE):" + str(rmse) + "\nMean (MAE: " + str(mae) +
                      " | Standard Deviation (MAE):" + str(std_mae) + "\n")

            except IndexError:
                print(text)
                sys.exit()

        else:
            print(text)
            sys.exit()

    except IndexError:
        print(text)
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
