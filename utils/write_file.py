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
