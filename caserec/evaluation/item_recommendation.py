# coding=utf-8
""""
    This class is responsible for evaluate item recommendation algorithms (rankings).

    This file contains item recommendation evaluation metrics:
        - Mean average precision - MAP
        - Precision
        - Recall
        - Normalized Discounted Cumulative Gain - NDCG

    Types of evaluation:
        - Simple: Evaluation with traditional strategy
        - All-but-one Protocol: Considers only one pair (u, i) from the test set to evaluate the ranking

"""

# © 2018. Case Recommender (MIT License)

import numpy as np
import random

from caserec.evaluation.base_evaluation import BaseEvaluation
from caserec.evaluation.item_recomendation_functions import precision_at_k, mean_average_precision, ndcg_at_k

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class ItemRecommendationEvaluation(BaseEvaluation):
    def __init__(self, sep='\t', n_ranks=list([1, 3, 5, 10]),
                 metrics=list(['PREC', 'RECALL', 'MAP', 'NDCG']), all_but_one_eval=False,
                 verbose=True, as_table=False, table_sep='\t'):
        """
        Class to evaluate predictions in a item recommendation (ranking) scenario

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param n_ranks: List of positions to evaluate the ranking
        :type n_ranks: list, default [1, 3, 5, 10]

        :param metrics: List of evaluation metrics
        :type metrics: list, default ('PREC', 'RECALL', 'MAP', 'NDCG')

        :param all_but_one_eval: If True, considers only one pair (u, i) from the test set to evaluate the ranking
        :type all_but_one_eval: bool, default False

        :param verbose: Print the evaluation results
        :type verbose: bool, default True

        :param as_table: Print the evaluation results as table (only work with verbose=True)
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        metrics = [m + '@' + str(n) for m in metrics for n in n_ranks]
        super(ItemRecommendationEvaluation, self).__init__(sep=sep, metrics=metrics, all_but_one_eval=all_but_one_eval,
                                                           verbose=verbose, as_table=as_table, table_sep=table_sep)

        self.n_ranks = n_ranks

    def evaluate(self, predictions, test_set):
        """
        Method to calculate all the metrics for item recommendation scenario using dictionaries of ranking
        and test set. Use read() in ReadFile to transform your file in a dict

        :param predictions: Dictionary with ranking information
        :type predictions: dict

        :param test_set: Dictionary with test set information.
        :type test_set: dict

        :return: Dictionary with all evaluation metrics and results
        :rtype: dict

        """

        eval_results = {}
        num_user = len(test_set['users'])
        partial_map_all = None

        if self.all_but_one_eval:
            for user in test_set['users']:
                # select a random item
                test_set['items_seen_by_user'][user] = [random.choice(test_set['items_seen_by_user'].get(user, [-1]))]

        for i, n in enumerate(self.n_ranks):
            if n < 1:
                raise ValueError('Error: N must >= 1.')

            partial_precision = list()
            partial_recall = list()
            partial_ndcg = list()
            partial_map = list()

            for user in test_set['users']:
                hit_cont = 0
                # Generate user intersection list between the recommended items and test.
                list_feedback = set(list(predictions.get(user, []))[:n])
                intersection = list(list_feedback.intersection(test_set['items_seen_by_user'].get(user, [])))

                if len(intersection) > 0:
                    ig_ranking = np.zeros(n)
                    for item in intersection:
                        hit_cont += 1
                        ig_ranking[list(predictions[user]).index(item)] = 1

                    partial_precision.append(precision_at_k([ig_ranking], n))
                    partial_recall.append((float(len(intersection)) / float(len(test_set['items_seen_by_user'][user]))))
                    partial_map.append(mean_average_precision([ig_ranking]))
                    partial_ndcg.append(ndcg_at_k(list(ig_ranking)))

                partial_map_all = partial_map

            # create a dictionary with final results
            eval_results.update({
                'PREC@' + str(n): round(sum(partial_precision) / float(num_user), 6),
                'RECALL@' + str(n): round(sum(partial_recall) / float(num_user), 6),
                'NDCG@' + str(n): round(sum(partial_ndcg) / float(num_user), 6),
                'MAP@' + str(n): round(sum(partial_map) / float(num_user), 6),
                'MAP': round(sum(partial_map_all) / float(num_user), 6)

            })

        if self.verbose:
            self.print_results(eval_results)

        return eval_results
