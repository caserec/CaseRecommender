# CaseRecommender - A Recommender Framework for Python
Case Recommender is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.  The framework aims to provide a rich set of components from which you can construct a customized recommender system from a set of algorithms. Case Recommender has different types of item recommendation and rating prediction approaches, and different metrics validation and evaluation.

# Algorithms
Item Recommedation:

- BPR MF

- Item KNN

- Item Attribute KNN

- User KNN

- User Attribute KNN

- Enseble BPR Learning

Rating Prediction:

- Matriz Factorization 

- BPR MF

- Item KNN

- Item Attribute KNN

- User KNN

# Evaluation and Validation Metrics

- All-but-one Protocoll

- Cross-fold- Validation

- Item Recommendation: Precision, Recall and Map

- Rating Prediction: MAE and RMSE


# Usage
Soon...

#Input

The input-files of traditional have to be placed in the corresponding subdirectory and are in csv-format with 3 columns 

- User

- Item

- Feedback

Example: user_1\titem_1\tfeedback

# Help CaseRecommender
To help the project with contribuitions follow the steps:

- Fork CaseRecommender

- Make your alterations and commit

- Create a topic branch - git checkout -b my_branch

- Push to your branch - git push origin my_branch

- Create a Pull Request from your branch.

- You just contributed to the CaseRecommender project!

For bugs or feedback use this link: https://github.com/ArthurFortes/CaseRecommender/issues

# Requirements 

- scipy
- numpy 

# Developed By

Arthur Fortes da Costa

University of São Paulo - ICMC (USP)

fortes.arthur@gmail.com

# LICENCE (GPL)


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
