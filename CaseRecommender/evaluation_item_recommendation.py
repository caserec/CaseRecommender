import getopt
import sys
import time
from CaseRecommender.evaluation.item_recommendation import ItemRecommendationEvaluation

__author__ = 'Arthur Fortes'

text = '____________________________________________________________________\n' \
       '\n  [Case Recommender Instructions] Evaluation Item Recommendation  \n' \
       '____________________________________________________________________\n\n' \
       'Evaluation Metrics:\n' \
       '  - Precision at (Prec@N)\n' \
       '  - Mean Average Precision at (MAP@N)\n' \
       '  - Mean Average Precision (MAP) \n\n' \
       ' * Protocol All-but-one \n' \
       '\nCommand: \n' \
       '  >> python evaluation_item_recommendation.py \n' \
       '\nArguments:\n' \
       '  -h -> HELP\n' \
       '  -e or --evaluation_type=  -> Values: SimpleEvaluation | AllButOne | FoldsEvaluation\n' \
       '  -s or --space_type=       -> Values: tabulation | comma | dot - Default Value: tabulation \n' \
       '  -r or --n_rank=           -> Number of evaluation rank positions - Default Values: 1,3,5,10 \n' \
       '\nIF -e SimpleEvaluation and AllButOne:\n' \
       '  -p or --predict_file=   -> prediction file with its directory \n' \
       '  -t or --test_file=      -> test file with its directory \n' \
       '\nIF -e FoldsEvaluation:\n' \
       '  -p or --predict_file=   -> only prediction file name without its directory \n' \
       '  -t or --test_file=      -> only test file name without its directory \n' \
       '  -d or --dir_fold=       -> folds directory \n' \
       '  -n or --num_fold=       -> number of folds - Default Value: 10\n' \
       '  -c or --type_fold_evaluation= -> Values: SimpleEvaluation | AllButOne' \
       ' - Default Value: SimpleEvaluation \n' \
       '\nExamples: \n' \
       '  >> python evaluation_item_recommendation.py -e SimpleEvaluation -p home\\documents\\file.dat' \
       '-t home\\documents\\test.dat -r 5,10\n' \
       '  >> python evaluation_item_recommendation.py -e AllButOne -p home\\documents\\file.dat' \
       '-t home\\documents\\test.dat -r 5,10 -s comma \n' \
       '  >> python evaluation_item_recommendation.py -e FoldsEvaluation -d home\\documents\\ -n 5 -c AllButOne ' \
       '-p file.dat -t test.dat -r 5,10 -s comma \n' \
       '  >> python evaluation_item_recommendation.py -e FoldsEvaluation -d home\\documents\\ -n 10 ' \
       '-p file.dat -t test.dat -r 3,5,10 -s dot \n' \
       '\nDetails: \n' \
       '  * In -r or --n_rank do not put space between commas. Wrong: -r 1, 3, 10 | Correct: 1,3,10\n'


def main(argv):
    evaluation_type = ''
    predict_file = ''
    test_file = ''
    dir_fold = ''
    num_fold = 10
    tfe = ''
    n_rank = [1, 3, 5, 10]
    space_type = '\t'

    try:
        opts, args = getopt.getopt(argv, "h:e:p:t:d:n:c:r:s:",
                                   ["evaluation_type=", "predict_file=", "test_file=", "dir_fold=", "num_fold=",
                                    "type_fold_evaluation=", "n_rank=", "space_type="])
    except getopt.GetoptError:
        print(text)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(text)
            sys.exit()
        elif opt in ("-s", "--space_type"):
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
        elif opt in ("-e", "--evaluation_type"):
            evaluation_type = arg
        elif opt in ("-p", "--predict_file"):
            predict_file = arg
        elif opt in ("-t", "--test_file"):
            test_file = arg
        elif opt in ("-d", "--dir_fold"):
            dir_fold = arg
        elif opt in ("-n", "--num_fold"):
            num_fold = arg
        elif opt in ("-c", "--type_fold_evaluation"):
            tfe = arg
        elif opt in ("-r", "--n_rank"):
            n_rank = arg
            new_list = list()
            for n in n_rank.split(','):
                new_list.append(int(n))
            n_rank = new_list

    if evaluation_type == "SimpleEvaluation":
        print("\n[Case Recommender - Evaluation Item Recommendation]")
        print "Evaluation type: ", evaluation_type
        print "Predict file: ", predict_file
        print "Test file: ", test_file
        print("Top Rank: " + str(n_rank) + "\n")
        starting_point = time.time()
        result = ItemRecommendationEvaluation(space_type, n_ranks=n_rank)
        list_results = result.simple_evaluation(predict_file, test_file)
        elapsed_time = time.time() - starting_point
        print("Runtime: " + str(elapsed_time / 60) + " minute(s)\n")
        list_labels = list()

        for i in n_rank:
            list_labels.append("Prec@" + str(i))
            list_labels.append("Recall@" + str(i))
        list_labels.append("MAP")

        for n, res in enumerate(list_results):
            print(str(list_labels[n] + ': ' + str(res)))
        print("\n")

    elif evaluation_type == "AllButOne":
        print("\n[Case Recommender - Evaluation Item Recommendation]")
        print "Evaluation type: ", evaluation_type
        print "Predict file: ", predict_file
        print "Test file: ", test_file
        print("Top Rank: " + str(n_rank) + "\n")
        starting_point = time.time()
        result = ItemRecommendationEvaluation(space_type)
        list_results = result.all_but_one_evaluation(predict_file, test_file, n_rank)
        elapsed_time = time.time() - starting_point
        print("Runtime: " + str(elapsed_time / 60) + " minute(s)\n")

        list_labels = list()
        for i in n_rank:
            list_labels.append("Prec@" + str(i))
            list_labels.append("Recall@" + str(i))
            list_labels.append("MAP@" + str(i))

        for n, res in enumerate(list_results):
            print(str(list_labels[n] + ': ' + str(res)))

        print("\n")

    elif evaluation_type == "FoldsEvaluation":
        print("\n[Case Recommender - Evaluation Item Recommendation]")
        print "Evaluation type: ", evaluation_type
        print "Predict file: ", predict_file
        print "Test file: ", test_file
        print("Folds Path: " + str(dir_fold))
        print("Number of Folds: " + str(num_fold))
        if tfe == '':
            tfe = 'SimpleEvaluation'
        print("Type Fold Evaluation: " + str(tfe))
        print("Top Rank: " + str(n_rank) + "\n")
        starting_point = time.time()
        result = ItemRecommendationEvaluation(space_type)
        list_results = result.folds_evaluation(dir_fold, int(num_fold), predict_file, test_file, tfe, n_rank)
        elapsed_time = time.time() - starting_point
        print("Runtime: " + str(elapsed_time / 60) + " minute(s)\n")

        list_labels = list()
        for i in n_rank:
            list_labels.append("Prec@" + str(i))
            list_labels.append("Recall@" + str(i))
            if tfe == "AllButOne":
                list_labels.append("MAP@" + str(i))
        if tfe == "SimpleEvaluation":
            list_labels.append("MAP")

        for n, res in enumerate(list_results):
            print(list_labels[n])
            print("Mean: " + str(res[0]))
            print("STD: " + str(res[1]))
        print("\n")

    elif evaluation_type == "":
        print(text)

    else:
        print('\nError: Invalid evaluation type!\n')
        print(text)
        sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
