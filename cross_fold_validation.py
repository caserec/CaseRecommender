import getopt
import sys

import time

from utils.cross_fold_validation import CrossFoldValidation

__author__ = 'Arthur Fortes'

text = '__________________________________________________________________\n' \
       '\n  [Case Recommender Instructions] Cross Fold Validation  \n' \
       '__________________________________________________________________\n\n'\
       '\nCommand: \n' \
       '\t python evaluation_rating_prediction.py \n' \
       '\t\t Arguments:\n' \
       '\t\t\t -h -> HELP\n' \
       '\t\t\t -d or --dataset=     -> Dataset with directory \n' \
       '\t\t\t -f or --dataset=     -> Directory where will be writing the folds \n' \
       '\t\t\t -n or --num_fold=    -> Number of folds \n' \
       '\t\t\t -s or --space_type=  -> Values: tabulation | comma | dot - Default Value: tabulation \n' \
       '\nExamples: \n ' \
       '\t >> python cross_fold_validation.py -d home\\documents\\file.dat -f home\\documents\\ -n 5' \
       ' -s comma\n'


def main(argv):
    dataset = ''
    dir_fold = ''
    num_fold = 10
    space_type = '\t'

    try:
        opts, args = getopt.getopt(argv, "h:d:f:n:s:",
                                   ["dataset=", "dir_fold=", "num_fold=", "space_type="])
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
        elif opt in ("-d", "--dataset="):
            dataset = arg
        elif opt in ("-f", "--dir_fold="):
            dir_fold = arg
        elif opt in ("-n", "--num_fold="):
            num_fold = arg

    if dataset == '':
        print(text)
        sys.exit()

    if dir_fold == '':
        print("\nError: Please enter a directory to write folds!\n")
        print(text)
        sys.exit()

    print("\n[Case Recommender - Cross Fold Validation]")
    print "Dataset File: ", dataset
    print "Dir Folds: ", dir_fold
    print "Number of Folds: ", num_fold
    print("\nPlease wait few seconds...")
    starting_point = time.time()
    CrossFoldValidation(dataset, space_type=space_type, dir_folds=dir_fold, n_folds=num_fold)
    elapsed_time = time.time() - starting_point
    print("Runtime: " + str(elapsed_time / 60) + " second(s)")
    print("Cross Fold Validation Finished!\n")


if __name__ == "__main__":
    main(sys.argv[1:])

# dataset, space_type='\t', dir_folds='', n_folds=10)
