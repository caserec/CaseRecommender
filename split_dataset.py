import getopt
import sys

import time

from utils.cross_fold_validation import CrossFoldValidation
from utils.split_dataset import SplitDataset

__author__ = 'Arthur Fortes'

text = '__________________________________________________________________\n' \
       '\n  [Case Recommender Instructions] Split Dataset  \n' \
       '__________________________________________________________________\n\n'\
       '\nCommand: \n' \
       '\t python split_dataset.py \n' \
       '\t\t Arguments:\n' \
       '\t\t\t -h -> HELP\n' \
       '\t\t\t -t or --split_type=  -> Values: CrossFoldValidation | SimpleSplit \n' \
       '\t\t\t -f or --dataset=     -> Directory where will be writing the folds \n' \
       '\t\t\t -n or --num_fold=    -> Number of folds \n' \
       '\t\t\t -s or --space_type=  -> Values: tabulation | comma | dot - Default Value: tabulation \n' \
       '\t\t\t * IF -t CrossFoldValidation:\n' \
       '\t\t\t\t -d or --dataset=     -> Dataset with directory [Accepts one file only] \n' \
       '\t\t\t * IF -t SimpleSplit:\n' \
       '\t\t\t\t -d or --dataset=     -> List of feedback types [Accepts one or more files in a list] \n' \
       '\t\t\t\t -r or --test_ratio=  -> Percentage of interactions dedicated to test set [Default = 0.2 = 20%] \n' \
       '\nExamples: \n ' \
       '\t >> python split_dataset.py -t CrossFoldValidation -d home\\documents\\file.dat -f home\\documents\\ -n 5' \
       ' -s comma\n' \
       '\t >> python split_dataset.py -t SimpleSplit -d [home\\documents\\rate.dat, home\\documents\\rate.dat] ' \
       '-f home\\documents\\ -n 5 -s comma -r 0.1\n'


def main(argv):
    dataset = ''
    dir_fold = ''
    num_fold = 10
    space_type = '\t'
    split_type = ''
    test_ratio = 0.2

    try:
        opts, args = getopt.getopt(argv, "h:t:d:f:n:s:r:",
                                   ["dataset=", "split_type", "dir_fold=", "num_fold=", "space_type=", "test_ratio="])
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
        elif opt in ("-t", "--split_type="):
            split_type = arg
        elif opt in ("-r", "--test_ratio="):
            test_ratio = arg

    if dataset == '':
        print(text)
        sys.exit()

    if split_type == '':
        print(text)
        sys.exit()

    if dir_fold == '':
        print("\nError: Please enter a directory to write folds!\n")
        print(text)
        sys.exit()

    print("\n[Case Recommender - Cross Fold Validation]")
    print "Dataset File(s): ", dataset
    print "Dir Folds: ", dir_fold
    print "Number of Folds: ", num_fold
    print("\nPlease wait few seconds...")
    starting_point = time.time()

    if split_type == 'CrossFoldValidation':
        CrossFoldValidation(dataset, space_type=space_type, dir_folds=dir_fold, n_folds=num_fold)
    elif split_type == 'SimpleSplit':
        print "Split dataset in train: ", (1-test_ratio) * 100, "% and test: ", test_ratio*100, "%"
        SplitDataset(dataset, space_type=space_type, dir_folds=dir_fold, n_folds=num_fold, test_ratio=test_ratio)
    else:
        print(text)
        sys.exit()
    elapsed_time = time.time() - starting_point
    print("Runtime: " + str(elapsed_time / 60) + " second(s)")
    print("SplitDataset Finished!\n")


if __name__ == "__main__":
    main(sys.argv[1:])
