# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file contains Statical functions for recommender systems.
    - T-test
    - Wilcoxon
"""

from scipy.stats import ttest_ind, ranksums
import numpy as np

__author__ = 'Arthur Fortes'


class StatisticalAnalysis(object):
    def __init__(self, sample1, sample2, method='ttest'):
        self.sample1 = np.array(sample1)
        self.sample2 = np.array(sample2)
        self.method = method

    def general_analysis(self):
        """
        Analyzing the difference

        Instead you might compute the difference and apply some common measure like the sum of absolute differences 
        (SAD), the sum of squared differences (SSD) or the correlation coefficient:
        """

        print("=== Information About Samples ===")
        print("Standard Deviation Sample1: " + str(np.std(self.sample1)))
        print("Standard Deviation Sample2: " + str(np.std(self.sample2)) + "\n")
        print("=== Analyzing the Difference Between Samples ===")
        print("SAD:" + str(np.sum(np.abs(self.sample1 - self.sample2))))
        print("SSD:" + str(np.sum(np.square(self.sample1 - self.sample2))))
        print("Correlation:" + str(np.corrcoef(np.array((self.sample1, self.sample2)))[0, 1]) + "\n")

    def ttest(self):
        """
        T-student
    
        Calculates the T-test for the means of TWO INDEPENDENT samples of scores.
    
        This is a two-sided test for the null hypothesis that 2 independent samples have identical 
        average (expected) values
    
        This test assumes that the populations have identical variances.
        """

        t, p = ttest_ind(self.sample1, self.sample2)
        print("=== T- Student Analysis ===")
        print("The calculated t-statistic: " + str(t))
        print("The two-tailed p-value: " + str(p) + "\n")

    def wilcoxon(self):
        """
        Wilcoxon
        
        The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from 
        the same distribution. In particular, it tests whether the distribution of the differences x - y 
        is symmetric about zero. It is a non-parametric version of the paired T-test.
        """

        t, p = ranksums(self.sample1, self.sample2)
        print("=== Wilcoxon Analysis ===")
        print("The calculated t-statistic: " + str(t))
        print("The two-tailed p-value: " + str(p) + "\n")

    def execute(self):
        self.general_analysis()
        if self.method.lower() == "wilcoxon":
            self.wilcoxon()
        elif self.method.lower() == "ttest":
            self.ttest()
        else:
            print("Error: Method Invalid!")
