# Case Recommender - A Recommender Framework for Python
Case Recommender is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.  The framework aims to provide a rich set of components from which you can construct a customized recommender system from a set of algorithms. Case Recommender has different types of item recommendation and rating prediction approaches, and different metrics validation and evaluation.

# Algorithms
Item Recommendation:

- BPR MF

- Item KNN

- Item Attribute KNN

- User KNN

- User Attribute KNN

- Ensemble BPR Learning

- Most Popular

- Random

Rating Prediction:

- Matrix Factorization (with and without baseline)

- SVD ++

- Item KNN

- Item Attribute KNN

- User KNN

- User Attribute KNN

- Item NSVD1 (with and without Batch)

- User NSVD1 (with and without Batch)


# Evaluation and Validation Metrics

- All-but-one Protocol

- Cross-fold- Validation

- Item Recommendation: Precision, Recall, NDCG and Map

- Rating Prediction: MAE and RMSE

- Statistical Analysis (T-test and Wilcoxon)

# Requirements 

- Python 3
- scipy
- numpy 

For Linux and MAC use:

    $ pip install requeriments

For Windows use:

    http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib
    

# Quick start

Case Recommender can be installed using pip:

    $ pip install caserecommender

If you want to run the latest version of the code, you can install from git:
    
    $ pip install -U git+git://github.com/ArthurFortes/CaseRecommender.git
    
# Usage

Divide datasets (Cross Fold Validation)

    >> from caserec.utils.cross_fold_validation import CrossFoldValidation
    >> CrossFoldValidation(input_file=dataset, dir_folds=dir_path, n_folds=10).execute()

Run Item Recommendation Algorithm (E.g: ItemKNN)

    >> from caserec.recommenders.item_recommendation.itemknn import ItemKNN
    >> ItemKNN(train_file, test_file).execute()
    
Run Rating Prediction Algorithm (E.g: ItemKNN)

    >> from caserec.recommenders.rating_prediction.itemknn import ItemKNN
    >> ItemKNN(train_file, test_file).execute()

Evaluate Ranking (Prec@N, Recall@N, NDCG@, Map@N and Map Total)

    >> from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
    >> ItemRecommendationEvaluation().simple_evaluation(ranking_file, test_file)
    
Evaluate Ranking (MAE and RMSE)

    >> from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
    >> RatingPredictionEvaluation().simple_evaluation(predictions_file, test_file)

# Documentation
Soon...

#Input

The input-files of traditional have to be placed in the corresponding subdirectory and are in csv-format with 3 columns 

- User

- Item

- Feedback

Example: user_1\titem_1\tfeedback

# Help CaseRecommender
To help the project with contributions follow the steps:

- Fork CaseRecommender

- Make your alterations and commit

- Create a topic branch - git checkout -b my_branch

- Push to your branch - git push origin my_branch

- Create a Pull Request from your branch.

- You just contributed to the CaseRecommender project!

For bugs or feedback use this link: https://github.com/ArthurFortes/CaseRecommender/issues

# Developed By

Arthur Fortes da Costa

University of São Paulo - ICMC (USP)

fortes.arthur@gmail.com

# Reference 

da Costa Fortes, A. and Manzato, M. G. Case recommender: A recommender framework. In Proceedings of the
22Nd Brazilian Symposium on Multimedia and the Web. Webmedia ’16. SBC, pp. 99–102, 2016.

BibTex (.bib)

    @inproceedings{daCostaCase:16,
    author = {da Costa Fortes, Arthur and Manzato, Marcelo Garcia},
    title = {Case Recommender: A Recommender Framework},
    booktitle = {Proceedings of the 22Nd Brazilian Symposium on Multimedia and the Web},
 	series = {Webmedia '16},
 	year = {2016},
    pages = {99--102},
    numpages = {4},
    publisher = {SBC}
    } 

# License (GPL)

    © 2017. Case Recommender All Rights Reserved

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
