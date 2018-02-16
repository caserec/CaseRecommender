"""

    Running item recommendation algorithms

"""
from caserec.recommenders.item_recommendation.bprmf import BprMF

tr = '/home/fortesarthur/Documentos/dataset/ml-100k/folds/0/train.dat'
te = '/home/fortesarthur/Documentos/dataset/ml-100k/folds/0/test.dat'


BprMF(tr, te, batch_size=30).compute()
