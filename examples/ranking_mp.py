"""
    Running Most Popular Recommender [Item Recommendation]

    - Cross Validation
    - Simple

"""

from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.utils.cross_validation import CrossValidation

db = '/home/fortesarthur/Documentos/dataset/hetrec2011-movielens-2k-v2/user_ratedmovies.dat'
folds_path = '/home/fortesarthur/Documentos/dataset/hetrec2011-movielens-2k-v2/'

tr = '/home/fortesarthur/Documentos/dataset/hetrec2011-movielens-2k-v2/folds/0/train.dat'
te = '/home/fortesarthur/Documentos/dataset/hetrec2011-movielens-2k-v2/folds/0/test.dat'

# Cross Validation
recommender = MostPopular(as_binary=True)

CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1).compute()

# Simple
MostPopular(tr, te, as_binary=True).compute()
