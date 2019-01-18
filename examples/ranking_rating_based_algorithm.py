"""
    Running Precision and Recall metrics on rating-based algorithms

"""

from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.nnmf import NNMF
from caserec.utils.process_data import ReadFile
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation

tr = '../../datasets/ml-100k/folds/0/train.dat'
te = '../../datasets/ml-100k/folds/0/test.dat'

# File to be saved model's predictions
predictions_output_filepath = './predictions_output.dat'

# Creating model and computing train / test sets
# model = MatrixFactorization(tr, te, output_file = predictions_output_filepath)
model = NNMF(tr, te, output_file = predictions_output_filepath)

model.compute(verbose=False)

# Using ReadFile class to read predictions from file
reader = ReadFile(input_file=predictions_output_filepath)
predictions = reader.read()

# Creating evaluator with item-recommendation parameters
evaluator = RatingPredictionEvaluation(sep = '\t', n_rank = [10], as_rank = True, metrics = ['PREC'])

# Getting evaluation
item_rec_metrics = evaluator.evaluate(predictions['feedback'], model.test_set)

print ('\nItem Recommendation Metrics:\n', item_rec_metrics)

model.predict()

print ('\nOriginal Rating Prediction Metrics:\n', model.evaluation_results)