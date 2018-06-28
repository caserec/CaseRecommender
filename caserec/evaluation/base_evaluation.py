# coding=utf-8
""""
    This class is base for evaluation strategies

    Types of evaluation:
        - Simple: Evaluation with traditional strategy
        - All-but-one Protocol: Considers only one pair (u, i) from the test set to evaluate the ranking

"""

# © 2018. Case Recommender (MIT License)

from collections import defaultdict

from caserec.utils.process_data import ReadFile

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class BaseEvaluation(object):
    def __init__(self, sep='\t', metrics=None, all_but_one_eval=False, verbose=True, as_table=False, table_sep='\t'):
        """
        Class to be base for evaluation strategies

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param metrics: List of evaluation metrics
        :type metrics: list, default None

        :param all_but_one_eval: If True, considers only one pair (u, i) from the test set to evaluate the ranking
        :type all_but_one_eval: bool, default False

        :param verbose: Print the evaluation results
        :type verbose: bool, default True

        :param as_table: Print the evaluation results as table (only work with verbose=True)
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """
        self.sep = sep
        self.all_but_one_eval = all_but_one_eval
        self.metrics = metrics
        self.verbose = verbose
        self.as_table = as_table
        self.table_sep = table_sep

    def evaluate(self, predictions, test_set):
        """
        Method to be implemented for each strategy using their respective metrics.
        Use read() in ReadFile to transform your file in a dict

        :param predictions: Dictionary with ranking information
        :type predictions: dict

        :param test_set: Dictionary with test set information.
        :type test_set: dict

        """
        raise NotImplemented

    def evaluate_with_files(self, prediction_file, test_file):
        """
        Method to evaluate predictions using files

        :param prediction_file: Predictions file with at least 2 columns for item recommendation
        (eg. user item [score (optional)]) and 3 columns for rating prediction (eg. user item rating)
        :type prediction_file: str

        :param test_file: Test file
        :type test_file: str

        :return: Dictionary with all evaluation metrics and results
        :rtype: dict

        """

        predict = ReadFile(prediction_file, sep=self.sep).read()
        test_set = ReadFile(test_file, sep=self.sep).read()

        return self.evaluate(predict['feedback'], test_set)

    def evaluate_recommender(self, predictions, test_set):
        """
        Method to evaluate recommender results. This method should be called by item recommender algorithms

        :param predictions: List with recommender output. e.g. [[user, item, score], [user, item2, score] ...]
        :type predictions: list

        :param test_set: Dictionary with test set information.
        :type test_set: dict

        :return: Dictionary with all evaluation metrics and results
        :rtype: dict

        """

        predictions_dict = {}

        for sample in predictions:
            predictions_dict.setdefault(sample[0], {}).update({sample[1]: sample[2]})

        return self.evaluate(predictions_dict, test_set)

    def evaluate_folds(self, folds_dir, predictions_file_name, test_file_name, k_folds=10):
        """
        Evaluate ranking in a set of folds. The name of folds needs to be integer and start with 0. e.g.
        Exist a dir '/home/user/folds', in which contains folds 0, 1, ..., 10.

        :param folds_dir: Directory of folds
        :type folds_dir: str

        :param k_folds: Number of folds
        :type k_folds: int, default 10

        :param predictions_file_name: Name of the ranking file
        :type predictions_file_name: str

        :param test_file_name: Name of the test file
        :type test_file_name: str

        :return: Dictionary with all evaluation metrics and results
        :rtype: dict

        """

        folds_results = defaultdict()

        for fold in range(k_folds):
            predictions_file = folds_dir + str(fold) + '/' + predictions_file_name
            test_file = folds_dir + str(fold) + '/' + test_file_name

            for key, value in self.evaluate_with_files(predictions_file, test_file).items():
                folds_results[key] = folds_results.get(key, 0) + value

        folds_results = {k: round(v / k_folds, 6) for k, v in folds_results.items()}

        if self.verbose:
            self.print_results(folds_results)

        return folds_results

    def print_results(self, evaluation_results):
        """
        Method to print the results

        :param evaluation_results: Dictionary with results. e.g. {metric: value}
        :type evaluation_results: dict

        """

        if self.as_table:
            header = ''
            values = ''
            for metric in self.metrics:
                header += metric.upper() + self.table_sep
                values += str(evaluation_results[metric.upper()]) + self.table_sep
            print(header)
            print(values)

        else:
            evaluation = 'Eval:: '
            for metrics in self.metrics:
                evaluation += metrics.upper() + ': ' + str(evaluation_results[metrics.upper()]) + ' '
            print(evaluation)
