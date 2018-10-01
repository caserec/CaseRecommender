from caserec.recommenders.item_recommendation.content_based import ContentBased
from caserec.recommenders.item_recommendation.item_attribute_knn import ItemAttributeKNN

train = '../../datasets/ml-100k/folds/0/train.dat'
test = '../../datasets/ml-100k/folds/0/test.dat'
rank_cb = '../../datasets/ml-100k/folds/0/rank_cb.dat'
rank_attr = '../../datasets/ml-100k/folds/0/rank_attr.dat'
similarity = '../../datasets/ml-100k/folds/0/vsm.dat'
top_n = 10
metrics = ('PREC', 'RECALL', 'NDCG', 'MAP')

ItemAttributeKNN(train, test, similarity_file=similarity, output_file=rank_attr, rank_length=50).\
    compute(metrics=metrics, n_ranks=[10, 20, 50])
ContentBased(train, test, similarity_file=similarity, output_file=rank_cb, rank_length=50).\
    compute(metrics=metrics, n_ranks=[10, 20, 50])

