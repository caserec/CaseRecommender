# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Given a dataset, this function is responsible for creating a set of training and test.

Standard file for reading:

user /t item /t feedback /n

* To change the spacing in the file, change the space_type var [default = \t].

By default algorithm divides the base into 90% for training and 10% for test.
To modify this parameter, change "test_ratio".

"""

import random
from framework.utils.read_file import ReadFile
from framework.utils.write_file import WriteFile

__author__ = 'Arthur Fortes'


class SplitDataset(object):
    def __init__(self, dataset, space_type='\t', dir_folds='', n_folds=10, test_ratio=0.1):
        self.dataset = dataset
        self.n_folds = n_folds
        self.space_type = space_type
        self.test_ratio = test_ratio
        self.dict_folds = dict()
        self.dict_feedback_folds = dict()

        self.divide_dataset()

        if dir_folds != '':
            WriteFile(dir_folds, self.dict_folds, self.space_type).split_dataset(self.dict_feedback_folds, self.dataset)

    def divide_dataset(self):
        tp = ReadFile(self.dataset, space_type=self.space_type)
        tp.split_dataset()

        for fold in xrange(self.n_folds):
            dict_feedback = list()
            tp.triple_dataset = list(set(tp.triple_dataset))
            random.shuffle(tp.triple_dataset)
            sp = int((1-self.test_ratio) * len(tp.triple_dataset))
            train = tp.triple_dataset[:sp]
            test = tp.triple_dataset[sp:]
            train.sort()
            test.sort(key=lambda x: x[0])
            train_set = list()
            test_set = list()

            for i, feedback in enumerate(self.dataset):
                dict_individual = dict()

                for triple in train:
                    try:
                        dict_individual.setdefault(triple[0], {}).update(
                            {triple[1]: tp.individual_interaction[i][triple[0]][triple[1]]})
                        train_set.append([triple[0], triple[1], tp.individual_interaction[i][triple[0]][triple[1]]])
                    except KeyError:
                        pass

                for triple_test in test:
                    try:
                        test_set.append([triple_test[0], triple_test[1],
                                         tp.individual_interaction[i][triple_test[0]][triple_test[1]]])
                    except KeyError:
                        pass

                dict_feedback.append(dict_individual)

            self.dict_feedback_folds[fold] = dict_feedback
            self.dict_folds[fold] = {'train': train_set, 'test': test_set}
