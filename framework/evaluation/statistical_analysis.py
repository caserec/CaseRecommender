# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file contains Statical functions for recommender systems.
    - T-test
"""

from scipy.stats import ttest_ind
import numpy as np

__author__ = 'Arthur Fortes'


def statistical_analysis(sample1, sample2):

    sample1, sample2 = np.array(sample1), np.array(sample2)

    print("=== Information About Samples ===")
    print("Standard Deviation Sample1: " + str(np.std(sample1)))
    print("Standard Deviation Sample2: " + str(np.std(sample2)))

    """
    T-student

    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values

    This test assumes that the populations have identical variances.
    """

    t, p = ttest_ind(sample1, sample2)

    print("=== T- Student Analysis ===")
    print("The calculated t-statistic: " + str(t))
    print("The two-tailed p-value: " + str(p))

    """
    Analyzing the difference

    Instead you might compute the difference and apply some common measure like the sum of absolute differences (SAD),
    the sum of squared differences (SSD) or the correlation coefficient:
    """

    print("=== Analyzing the Difference Between Samples ===")
    print "SAD:", np.sum(np.abs(sample1 - sample2))
    print "SSD:", np.sum(np.square(sample1 - sample2))
    print "Correlation:", np.corrcoef(np.array((sample1, sample2)))[0, 1]
