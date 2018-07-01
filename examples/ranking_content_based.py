from caserec.recommenders.item_recommendation.content_based import ContentBased
from caserec.recommenders.item_recommendation.item_attribute_knn import ItemAttributeKNN

train = 'C:/datasets/ml-100k/train.dat'
test = 'C:/datasets/ml-100k/test.dat'
rank_cb = 'C:/datasets/ml-100k/rank_cb.dat'
rank_attr = 'C:/datasets/ml-100k/rank_attr.dat'
similarity = 'C:/datasets/ml-100k/vsm.dat'
top_n = 10
metrics = ('PREC', 'RECALL', 'NDCG', 'MAP')

ItemAttributeKNN(train, test, similarity_file=similarity, output_file=rank_attr, rank_length=50).\
    compute(metrics=metrics, n_ranks=[10, 20, 50])
ContentBased(train, test, similarity_file=similarity, output_file=rank_cb, rank_length=50).\
    compute(metrics=metrics, n_ranks=[10, 20, 50])

