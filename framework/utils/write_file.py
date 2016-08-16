# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file is responsible for write external files using data from tool.

The output file format is:
    user\titem\tinformation\n

* To change the spacing in the file, change the space_type var [default = \t].

Methods:
    - split_dataset: write train and test files for n folds and individuals feedback
    - cross_fold_validation: write train and test files for n folds.
    - write_prediction_file: write prediction file for rating prediction algorithms
    - write_ranking_file: write ranking file for item recommendation algorithms
    - write_ensemble: write ranking file for ensemble algorithms
"""

import os

__author__ = 'Arthur Fortes'


class WriteFile(object):
    def __init__(self, file_write, data, space_type="\t"):
        self.file_write = file_write
        self.data = data
        self.space_type = space_type

    def split_dataset(self, feedback=None, dataset=None):
        names = list()

        if dataset is not None:
            for t in dataset:
                if t.rfind('\\') == -1:
                    names.append(t[t.rfind('/')+1:])
                else:
                    names.append(t[t.rfind('\\')+1:])

        self.file_write += '\\folds\\'

        if not os.path.exists(self.file_write):
            os.mkdir(self.file_write)

        for f in self.data:
            fold_name = self.file_write + str(f) + '\\'
            if not os.path.exists(fold_name):
                os.mkdir(fold_name)

            with open(fold_name + 'train.dat', 'w') as infile:
                for triple in self.data[f]['train']:
                    infile.write(str(triple[0]) + self.space_type + str(triple[1]) +
                                 self.space_type + str(triple[2]) + '\n')

            with open(fold_name + 'test.dat', 'w') as infile:
                for triple in self.data[f]['test']:
                    infile.write(str(triple[0]) + self.space_type + str(triple[1]) +
                                 self.space_type + str(triple[2]) + '\n')

            if feedback is not None:
                for i, feed in enumerate(feedback[f]):
                    train_name = 'train_' + str(names[i])
                    with open(fold_name + train_name, 'w') as infile:
                        for user in feed:
                            for item in feed[user]:
                                infile.write(str(user) + self.space_type + str(item) + self.space_type +
                                             str(feed[user][item]) + '\n')

    def cross_fold_validation(self):
        self.file_write += '/folds/'

        if not os.path.exists(self.file_write):
            os.mkdir(self.file_write)

        # f is an integer number (0 - n fold)
        for f in self.data:
            fold_name = self.file_write + str(f) + '/'
            if not os.path.exists(fold_name):
                os.mkdir(fold_name)

            with open(fold_name + 'train.dat', 'w') as infile:
                for triple in self.data[f]['train']:
                    infile.write(str(triple[0]) + self.space_type + str(triple[1]) +
                                 self.space_type + str(triple[2]) + '\n')

            with open(fold_name + 'test.dat', 'w') as infile:
                for triple in self.data[f]['test']:
                    infile.write(str(triple[0]) + self.space_type + str(triple[1]) +
                                 self.space_type + str(triple[2]) + '\n')

    def write_prediction_file(self):
        self.data = sorted(self.data, key=lambda x: x[0])
        with open(self.file_write, "w") as infile:
            for t in self.data:
                infile.write(str(t[0]) + self.space_type + str(t[1]) + self.space_type + str(t[2]) + "\n")

    def write_ranking_file(self):
        with open(self.file_write, "w") as infile:
            for t in self.data:
                infile.write(str(t[0]) + self.space_type + str(t[1]) + self.space_type + str(t[2]) + "\n")

    def write_ensemble(self, list_users):
        with open(self.file_write, "w") as infile:
            for user in list_users:
                for t in self.data[user]:
                    infile.write(str(user) + self.space_type + str(t[0]) + self.space_type + str(t[1]) + "\n")
