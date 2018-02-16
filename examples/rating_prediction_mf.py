"""
    Running MF / SVD Recommenders [Rating Prediction]

    - Cross Validation
    - Simple

"""

from caserec.recommenders.rating_prediction.svdplusplus import SVDPlusPlus
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.utils.cross_validation import CrossValidation

db = 'C:/Users/forte/OneDrive/ml-100k/u.data'
folds_path = 'C:/Users/forte/OneDrive/ml-100k/'

metadata_item = 'C:/Users/forte/OneDrive/ml-100k/db_item_subject.dat'
sm_item = 'C:/Users/forte/OneDrive/ml-100k/sim_item.dat'
metadata_user = 'C:/Users/forte/OneDrive/ml-100k/metadata_user.dat'
sm_user = 'C:/Users/forte/OneDrive/ml-100k/sim_user.dat'

tr = 'C:/Users/forte/OneDrive/ml-100k/folds/0/train.dat'
te = 'C:/Users/forte/OneDrive/ml-100k/folds/0/test.dat'

"""

    UserKNN

"""

# Cross Validation
recommender = MatrixFactorization()

# CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

# # Simple
MatrixFactorization(tr, te).compute()
SVDPlusPlus(tr, te).compute()
