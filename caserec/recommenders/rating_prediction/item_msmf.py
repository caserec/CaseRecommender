# coding=utf-8
"""
    Item-MSMF: Items Most Similar based on Matrix Factorization
    [Rating Prediction]

    Literature:
        2018 Brazilian Conference on Intelligent Systems (BRACIS).
        Link soon.

"""

import numpy as np

from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.utils.extra_functions import timed

__author__ = 'Eduardo Fressato <eduardofressato@hotmail.com>'


class ItemMSMF(MatrixFactorization):
    def __init__(self, train_file=None, test_file=None, output_file=None, similarity_file=None, neighbors=20,
                 factors=10, learn_rate=0.01, epochs=30, delta=0.015, init_mean=0.1, init_stdev=0.1, baseline=True,
                 bias_learn_rate=0.005, delta_bias=0.002, stop_criteria=0.009, sep='\t', output_sep='\t',
                 similarity_sep='\t', random_seed=None, verbose=True):

        super(ItemMSMF, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file, sep=sep,
                                       learn_rate=learn_rate, factors=factors, epochs=epochs, delta=delta,
                                       init_mean=init_mean, init_stdev=init_stdev, baseline=baseline,
                                       bias_learn_rate=bias_learn_rate, delta_bias=delta_bias,
                                       stop_criteria=stop_criteria, output_sep=output_sep, random_seed=random_seed)

        """
            Item-MSMF: Items Most Similar based on Matrix Factorization

            The Item-MSMF algorithm, this is recommender technique based on matrix factorization, that incorporates
            similarities of items which are calculated based on metadata. This approach to address the item
            cold-start through a shared latent factor vector representation of similar items based on those items
            which have enough interactions with users. In this way, the new items representations that are not
            accurate in terms of rating prediction, is replaced them with a weighted average of the latent factor
            vectors of the most similar items.

            Usage::

                >> ItemMSMF(train, test, similarity_file, neighbors).compute()
                
            :param train_file: File which contains the train set. This file needs to have at least 3 columns
            (user item feedback_value).
            :type train_file: str
    
            :param test_file: File which contains the test set. This file needs to have at least 3 columns
            (user item feedback_value).
            :type test_file: str, default None
            
            :param output_file: File with dir to write the final predictions
            :type output_file: str, default None
            
            :param similarity_file: File which contains the similarity of items. This file needs to have at least 3 columns
            (item item similarity).
            :type similarity_file: str, default None
            
            :param neighbors: Number of items that replace the new item vector
            :type neighbors: int, default 20
    
            :param factors: Number of latent factors per user/item
            :type factors: int, default 10
    
            :param learn_rate: Learning rate (alpha)
            :type learn_rate: float, default 0.05
    
            :param epochs: Number of epochs over the training data
            :type epochs: int, default 30
    
            :param delta: Regularization value
            :type delta: float, default 0.015
    
            :param init_mean: Mean of the normal distribution used to initialize the latent factors
            :type init_mean: float, default 0
    
            :param init_stdev: Standard deviation of the normal distribution used to initialize the latent factors
            :type init_stdev: float, default 0.1
    
            :param bias_learn_rate: Learning rate for baselines
            :type bias_learn_rate: float, default 0.005
    
            :param delta_bias: Regularization value for baselines
            :type delta_bias: float, default 0.002
    
            :param stop_criteria: Difference between errors for stopping criteria
            :type stop_criteria: float, default 0.009
    
            :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
            :type random_seed: int, default None
            
            :param verbose: Print information
            :type verbose: bool, default True
                        
        """

        self.recommender_name = 'Item-MSMF for Cold Start'
        self.verbose = verbose

        self.similarity_file = similarity_file
        self.similarity_sep = similarity_sep
        self.si_matrix = None
        self.new_items = set()
        self.k = neighbors

    def init_model(self):
        super(ItemMSMF, self).init_model()
        if self.verbose:
            print("\nread_similarity_matrix_time:: %4f sec" % timed(self.fill_similarity_matrix))
        else:
            self.fill_similarity_matrix()

    def fill_similarity_matrix(self):
        self.si_matrix = np.zeros((len(self.items), len(self.items)))
        items_sim = set()

        with open(self.similarity_file, "r", encoding='utf-8') as infile:
            items = set(self.items)
            for line in infile:
                if line.strip():
                    inline = line.split(self.similarity_sep)
                    item_a, item_b, sim = int(inline[0]), int(inline[1]), float(inline[2].rstrip())

                    if item_a in items and item_b in items:
                        map_a = self.item_to_item_id[item_a]
                        map_b = self.item_to_item_id[item_b]
                        items_sim.add(item_a)
                        items_sim.add(item_b)
                        self.si_matrix[map_a][map_b] = sim
                        self.si_matrix[map_b][map_a] = sim

        if self.verbose:
            print("Number of item in similarity file:", len(items_sim))
        del items_sim

    def search_new_items(self):
        for i in self.test_set['items']:
            if i not in self.train_set['items']:
                self.new_items.add(i)

    def search_similar_items(self, item):
        item_index = self.item_to_item_id[item]
        count = 0
        list_items = []
        list_similar = sorted(enumerate(self.si_matrix[item_index]), key=lambda x: -x[1])

        for i, sim in list_similar:
            if i != item_index:
                if self.item_id_to_item[i] in self.train_set['items']:
                    list_items.append((self.item_id_to_item[i], sim))
                    count += 1
                    if count == self.k:
                        return list_items

    def replace_vector_new_item(self):

        for item in self.new_items:
            list_items = self.search_similar_items(item)

            q_i = self.q[self.item_to_item_id[list_items[0][0]]].copy() * list_items[0][1]
            b_i = self.bi[self.item_to_item_id[list_items[0][0]]].copy() * list_items[0][1]
            sum_sim = list_items[0][1]

            for item_j, sim in list_items[1:]:
                q_i += self.q[self.item_to_item_id[item_j]].copy() * sim
                b_i += self.bi[self.item_to_item_id[item_j]].copy() * sim
                sum_sim += sim

            if sum_sim > 0:
                q_i = q_i / sum_sim
                b_i = b_i / sum_sim

                self.q[self.item_to_item_id[item]] = q_i.copy()
                if self.baseline:
                    self.bi[self.item_to_item_id[item]] = b_i.copy()

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):

        if verbose:
            super(MatrixFactorization, self).compute(verbose=verbose)
            self.init_model()

            print("training_time:: %4f sec" % timed(self.fit))
            if self.extra_info_header is not None:
                print(self.extra_info_header)

            search_time = timed(self.search_new_items)
            replace_time = timed(self.replace_vector_new_item)
            prediction_time = timed(self.predict)

            print("search_new_items_time:: %4f sec" % search_time)
            print("vectors_replacement_time:: %4f sec" % replace_time)
            print("prediction_time:: %4f sec" % prediction_time)
            print("total_prediction_time:: %4f sec" % (search_time + replace_time + prediction_time))
            print("\n")

        else:
            # Execute all in silence without prints
            super(MatrixFactorization, self).compute(verbose=verbose)
            self.init_model()
            self.fit()
            self.search_new_items()
            self.replace_vector_new_item()
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            return self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)
