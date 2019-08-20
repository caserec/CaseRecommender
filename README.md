# Case Recommender - A Python Framework for RecSys

[![PyPI version](https://badge.fury.io/py/CaseRecommender.svg)](https://badge.fury.io/py/CaseRecommender)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![GitHub license](https://img.shields.io/github/license/caserec/CaseRecommender.svg)](https://github.com/caserec/CaseRecommender/blob/master/COPYING)

Case Recommender is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback. The framework aims to provide a rich set of components from which you can construct a customized recommender system from a set of algorithms. Case Recommender has different types of item recommendation and rating prediction approaches, and different metrics validation and evaluation.

# Algorithms

Item Recommendation:

- BPRMF

- ItemKNN

- Item Attribute KNN

- UserKNN

- User Attribute KNN

- Group-based (Clustering-based algorithm)

- Paco Recommender (Co-Clustering-based algorithm)

- Most Popular

- Random

- Content Based

Rating Prediction:

- Matrix Factorization (with and without baseline)

- Non-negative Matrix Factorization

- SVD

- SVD++

- ItemKNN

- Item Attribute KNN

- UserKNN

- User Attribute KNN

- Item NSVD1 (with and without Batch)

- User NSVD1 (with and without Batch)

- Most Popular

- Random

- gSVD++

- Item-MSMF

- (E) CoRec

Clustering:

- PaCo: EntroPy Anomalies in Co-Clustering

- k-medoids

# Evaluation and Validation Metrics

- All-but-one Protocol

- Cross-fold-Validation

- Item Recommendation: Precision, Recall, NDCG and Map

- Rating Prediction: MAE and RMSE

- Statistical Analysis (T-test and Wilcoxon)

# Requirements

- Python
- scipy
- numpy
- pandas
- scikit-learn

For Linux and MAC use:

    $ pip install requirements

For Windows use:

    http://www.lfd.uci.edu/~gohlke/pythonlibs/

# Installation

Case Recommender can be installed using pip:

    $ pip install caserecommender

If you want to run the latest version of the code, you can install from git:

    $ pip install -U git+git://github.com/caserec/CaseRecommender.git

# Quick Start and Guide

For more information about RiVal and the documentation, visit the Case Recommender [Wiki](https://github.com/caserec/CaseRecommender/wiki). If you have not used Case Recommender before, do check out the Getting Started guide.

# Usage

Divide Database (Fold Cross Validation)

    >> from caserec.utils.split_database import SplitDatabase
    >> SplitDatabase(input_file=dataset, dir_folds=dir_path, n_splits=10).k_fold_cross_validation()

Run Item Recommendation Algorithm (E.g: ItemKNN)

    >> from caserec.recommenders.item_recommendation.itemknn import ItemKNN
    >> ItemKNN(train_file, test_file).compute()

Run Rating Prediction Algorithm (E.g: ItemKNN)

    >> from caserec.recommenders.rating_prediction.itemknn import ItemKNN
    >> ItemKNN(train_file, test_file).compute()

Evaluate Ranking (Prec@N, Recall@N, NDCG@, Map@N and Map Total)

    >> from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
    >> ItemRecommendationEvaluation().evaluate_with_files(predictions_file, test_file)

Evaluate Ranking (MAE and RMSE)

    >> from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
    >> RatingPredictionEvaluation().evaluate_with_files(predictions_file, test_file)

# Input

The input-files of traditional have to be placed in the corresponding subdirectory and are in csv-format with at least
3 columns. Example: user_1,item_1,feedback

# Cite us

If you use Case Recommender in a scientific publication, we would appreciate citations of our paper where this framework was first mentioned and used.

To cite Case Recommender use: Arthur da Costa, Eduardo Fressato, Fernando Neto, Marcelo Manzato, and Ricardo Campello. 2019. Case recommender: a flexible and extensible python framework for recommender systems. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18). ACM, New York, NY, USA, 494-495. DOI: https://doi.org/10.1145/3240323.3241611.

For TeX/LaTeX (BibTex):

        @inproceedings{daCosta:2018:CRF:3240323.3241611,
            author = {da Costa, Arthur and Fressato, Eduardo and Neto, Fernando and Manzato, Marcelo and Campello, Ricardo},
            title = {Case Recommender: A Flexible and Extensible Python Framework for Recommender Systems},
            booktitle = {Proceedings of the 12th ACM Conference on Recommender Systems},
            series = {RecSys '18},
            year = {2018},
            isbn = {978-1-4503-5901-6},
            location = {Vancouver, British Columbia, Canada},
            pages = {494--495},
            numpages = {2},
            url = {http://doi.acm.org/10.1145/3240323.3241611},
            doi = {10.1145/3240323.3241611},
            acmid = {3241611},
            publisher = {ACM},
            address = {New York, NY, USA},
            keywords = {framework, python, recommender systems},
        }

# Help CaseRecommender

To help the project with contributions follow the steps:

- Fork CaseRecommender

- Make your alterations and commit

- Create a topic branch - git checkout -b my_branch

- Push to your branch - git push origin my_branch

- Create a Pull Request from your branch.

- You just contributed to the CaseRecommender project!

For bugs or feedback use this link: https://github.com/caserec/CaseRecommender/issues

# License (MIT)

    © 2019. Case Recommender All Rights Reserved

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
