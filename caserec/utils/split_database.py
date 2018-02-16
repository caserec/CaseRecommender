# coding=utf-8
""""
    This class is responsible for divide databases in k folds with two strategies:
        k-fold cross-validation or ShuffleSplit

"""

# © 2018. Case Recommender (MIT License)


from sklearn.model_selection import KFold, ShuffleSplit
import os

from caserec.utils.process_data import ReadFile, WriteFile

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class SplitDatabase(ReadFile):
    def __init__(self, input_file, dir_folds=None, n_splits=10, sep_read='\t', sep_write='\t', header=None,
                 names=None, as_binary=False, binary_col=None, write_mode='w'):
        """
        Given a database, this class is responsible for creating a training and test sets
        for k folds with well-known strategies:

        - k-fold cross-validation
        - ShuffleSplit

        Usage:

            >> SplitDatabase(input_file=database, dir_folds=dir_path, n_folds=10).kfoldcrossvalidation()
            >> SplitDatabase(input_file=database, dir_folds=dir_path, n_folds=10).shuffle_split(test_size=0.3)
            # To use only one fold, you should use only shuffle_split. kfoldcrossvalidation works only with
            # n_folds >= 2:
            >> SplitDatabase(input_file=database, dir_folds=dir_path, n_folds=1).shuffle_split(test_size=0.1)

        :param input_file: Input File with at least 2 columns.
        :type input_file: str

        :param dir_folds: Directory to write folds (train and test files)
        :type dir_folds: str

        :param n_splits: How much folds the strategy will divide
        :type n_splits: int, default 10

        :param sep_read: Delimiter for input files
        :type sep_read: str, default '\t'

        :param sep_write: Delimiter for output files
        :type sep_write: str, default '\t'

        :param header: Skip header line (only work with method: read_with_pandas)
        :type header: int, default None

        :param names: Name of columns (only work with method: read_with_pandas)
        :type names: str, default None

        :param as_binary: If True, the explicit feedback will be transform to binary
        :type as_binary: bool, default False

        :param binary_col: Index of columns to read as binary (only work with method: read_with_pandas)
        :type binary_col: int, default 2

        :param write_mode: Method to write file
        :type write_mode: str, default 'w'

        """

        super(SplitDatabase, self).__init__(input_file, sep=sep_read, header=header, names=names, as_binary=as_binary,
                                            binary_col=binary_col)

        self.dir_folds = dir_folds
        self.n_splits = n_splits
        self.sep_write = sep_write
        self.write_mode = write_mode
        self.df = self.read_with_pandas()

        if self.dir_folds is not None:
            self.create_folds()

    def create_folds(self):
        self.dir_folds += "folds/"
        if not os.path.exists(self.dir_folds):
            os.mkdir(self.dir_folds)

        for n in range(self.n_splits):
            if not os.path.exists(self.dir_folds + str(n)):
                os.mkdir(self.dir_folds + str(n))

    def write_files(self, trained_model):
        fold = 0
        for train_index, test_index in trained_model:
            if self.dir_folds is not None:
                train_file = self.dir_folds + str(fold) + '/train.dat'
                test_file = self.dir_folds + str(fold) + '/test.dat'

                df_train = self.df.ix[train_index]
                df_test = self.df.ix[test_index]

                WriteFile(train_file, sep=self.sep_write, mode=self.write_mode
                          ).write_with_pandas(df_train.sort_values(by=[0, 1]))
                WriteFile(test_file, sep=self.sep_write, mode=self.write_mode
                          ).write_with_pandas(df_test.sort_values(by=[0, 1]))

                fold += 1

    def kfoldcrossvalidation(self, shuffle=True, random_state=None):
        """
        k-fold cross-validation

        In k-fold cross-validation, the original sample is randomly partitioned into
        k equal sized subsamples. Of the k subsamples, a single subsample is retained as
        the validation data for testing the model, and the remaining k − 1 subsamples are
        used as training data. The cross-validation process is then repeated k times (the folds),
        with each of the k subsamples used exactly once as the validation data.

        The k results from the folds can then be averaged (or otherwise combined) to produce a
        single estimation. Reference: https://en.wikipedia.org/wiki/Cross-validation_(statistics)

        :param shuffle:
        :type shuffle:

        :param random_state:
        :type random_state:

        :return:
        """

        kfold = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        trained_model = list(kfold.split(self.df))

        if self.dir_folds is not None:
            self.write_files(trained_model)

        return trained_model

    def shuffle_split(self, test_size=0.1, random_state=None):
        """
        Shuffle Split

        Random permutation cross-validator

        Yields indices to split data into training and test sets.

        Note: contrary to other cross-validation strategies, random splits do not guarantee that
        all folds will be different, although this is still very likely for sizeable databases.

        :param test_size:
        :type test_size:

        :param random_state:
        :type random_state:

        :return:
        """
        ss = ShuffleSplit(n_splits=self.n_splits, test_size=test_size, random_state=random_state)
        trained_model = list(ss.split(self.df))

        if self.dir_folds is not None:
            self.write_files(trained_model)

        return trained_model
