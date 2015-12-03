import os

__author__ = 'Arthur Fortes'

'''
This file is responsible for write external files using data from tool.

The output file format is:

user\titem\tinformation\n

* To change the spacing in the file, change the space_type var [default = \t].

Methods:
    - [cross_fold_validation]: write train and test files for n folds.
'''


class WriteFile(object):
    def __init__(self, file_write, data, space_type='\t'):
        self.file_write = file_write
        self.data = data
        self.space_type = space_type

    def split_dataset(self, feedback='', dataset=''):
        names = list()

        if dataset != '':
            for t in dataset:
                if t.rfind('\\') == -1:
                    names.append(t[t.rfind('/')+1:])
                else:
                    names.append(t[t.rfind('\\')+1:])

        self.file_write += 'folds\\'

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

            if feedback != '':
                for i, feed in enumerate(feedback[f]):
                    train_name = 'train_' + str(names[i])
                    with open(fold_name + train_name, 'w') as infile:
                        for user in feed:
                            for item in feed[user]:
                                infile.write(str(user) + self.space_type + str(item) + self.space_type +
                                             str(feed[user][item]) + '\n')

    # We used the variable self.file_write as directory path in this function.
    def cross_fold_validation(self):
        self.file_write += 'folds\\'

        if not os.path.exists(self.file_write):
            os.mkdir(self.file_write)

        # f is an integer number (0 - n fold)
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

    # This function can be used to item recommendation and rating prediction
    def write_prediction_file(self):
        with open(self.file_write, 'w') as infile_write:
            for user in self.data:
                for item in user[1]:
                    infile_write.write(str(user[0]) + self.space_type + str(item[0]) + self.space_type +
                                       str(item[1]) + '\n')
