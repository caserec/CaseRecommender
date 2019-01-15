# coding=utf-8
""""
    These functions are responsible for evaluate item recommendation algorithms (rankings).

    They are used by evaluation/item_recommendation.py

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


def precision_at_k(ranking, k):
    """
    Score is precision @ k
    Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :param k: length of ranking
    :type k: int

    :return: Precision @ k
    :rtype: float

    """

    assert k >= 1
    ranking = np.asarray(ranking)[:k] != 0
    if ranking.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(ranking)


def average_precision(ranking):
    """
    Score is average precision (area under PR curve). Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :return: Average precision
    :rtype: float

    """

    ranking = np.asarray(ranking) != 0
    out = [precision_at_k(ranking, k + 1) for k in range(ranking.size) if ranking[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(ranking):
    """
    Score is mean average precision. Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :return: Mean average precision
    :rtype: float
    """

    return np.mean([average_precision(r) for r in ranking])


def ndcg_at_k(ranking):
    """
    Score is normalized discounted cumulative gain (ndcg). Relevance is positive real values.  Can use binary
    as the previous methods.

    :param ranking: ranking to evaluate in dcg format [0, 0, 1], where 1 is correct info
    :type ranking: list

    :return: Normalized discounted cumulative gain
    :rtype: float

    """

    ranking = np.asfarray(ranking)
    r_ideal = np.asfarray(sorted(ranking, reverse=True))
    dcg_ideal = r_ideal[0] + np.sum(r_ideal[1:] / np.log2(np.arange(2, r_ideal.size + 1)))
    dcg_ranking = ranking[0] + np.sum(ranking[1:] / np.log2(np.arange(2, ranking.size + 1)))

    return dcg_ranking / dcg_ideal
