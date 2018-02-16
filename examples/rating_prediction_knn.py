"""
    Running KNN Recommenders [Rating Prediction]

    - Cross Validation
    - Simple

"""

from caserec.recommenders.rating_prediction.user_attribute_knn import UserAttributeKNN
from caserec.recommenders.rating_prediction.item_attribute_knn import ItemAttributeKNN
from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.userknn import UserKNN
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
recommender = UserKNN()

CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

# # Simple
UserKNN(tr, te).compute()
# UserAttributeKNN(tr, te, metadata_file=metadata_user).compute()
# UserAttributeKNN(tr, te, similarity_file=sm_user).compute()

"""

    ItemKNN

"""

# # Cross Validation
recommender = ItemKNN()

CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()
#
# # Simple
ItemKNN(tr, te).compute()
# ItemAttributeKNN(tr, te, metadata_file=metadata_item).compute()
# ItemAttributeKNN(tr, te, similarity_file=sm_item).compute()
