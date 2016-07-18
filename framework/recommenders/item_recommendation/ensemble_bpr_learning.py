import math
import random

import numpy as np

from framework.utils import ReadFile
from framework.utils import WriteFile
from framework.utils import check_len_lists

__author__ = 'Arthur Fortes'

num_interactions = 30
num_interactions_baselines = 15000
num_factor = 30
num_interactions_train_weights = 10000
learn_rate = 0.05
reg_bias = 0
reg_bias_beta = 0.0025
reg_u = 0.0025
reg_i = 0.0025
reg_j = 0.0025


# utils
def sample_triple(dict_item, dict_not_item, list_users):
    user = random.choice(list_users)
    item = random.choice(dict_item[user].keys())
    other_item = random.choice(dict_not_item[user])
    return user, item, other_item


def create_factors(num_users, num_items, factors):
    users_factors = np.random.uniform(0, 1, [num_users, factors])
    items_factors = np.random.uniform(0, 1, [num_items, factors])
    bias = np.random.uniform(0, 1, num_items)
    return users_factors, items_factors, bias


def return_list_info(train_set):
    lu = set()
    li = set()
    dict_users_interactions = dict()
    dict_non_seen_items = dict()
    dict_index = dict()

    for interaction in train_set:
        user, item, score = interaction[0], interaction[1], interaction[2]
        lu.add(user)
        li.add(item)
        dict_users_interactions.setdefault(user, {}).update({item: score})

    for u, user in enumerate(lu):
        dict_index.setdefault('users', {}).update({user: u})
        dict_non_seen_items[user] = list(li - set(dict_users_interactions[user].keys()))

    for i, item in enumerate(li):
        dict_index.setdefault('items', {}).update({item: i})

    return dict_users_interactions, dict_non_seen_items, lu, li, dict_index


class EnsembleLearningBPR(object):
    def __init__(self, list_train_files, list_rank_files, file_write, rank_number=10, space_type='\t'):
        self.list_train_files = list_train_files
        self.list_rank_files = list_rank_files
        self.file_write = file_write
        self.rank_number = rank_number
        self.space_type = space_type
        check_len_lists(self.list_train_files, self.list_rank_files)
        self.num_interactions = len(self.list_train_files)
        self.factors = list()
        self.individual_datasets = list()
        self.final_dataset = list()
        self.betas = list()

        # vars
        self.dict_item = dict()
        self.dict_not_item = dict()
        self.list_users = set()
        self.list_items = set()
        self.dict_index = dict()
        self.rankings = list()
        self.final_ranking = list()
        self.normalization = list()

        # call internal methods
        self.read_ranking_files()
        print('Read Ranking Files...')
        self.treat_interactions()
        print('Trained baselines...')
        self.train_weights()
        print('Trained betas...')
        self.ensemble_ranks()
        print('Finished Ensemble interactions...')
        self.write_ranking()

    def read_ranking_files(self):
        for ranking_file in self.list_rank_files:
            ranking = ReadFile(ranking_file, space_type=self.space_type)
            rank_interaction, list_interaction = ranking.read_rankings()
            self.rankings.append(rank_interaction)
            self.normalization.append([min(list_interaction), max(list_interaction)])

    def treat_interactions(self):
        for interaction_file in self.list_train_files:
            interaction = ReadFile(interaction_file, space_type=self.space_type)
            interaction.triple_information()
            self.individual_datasets.append(interaction.triple_dataset)
            self.final_dataset += interaction.triple_dataset

        self.dict_item, self.dict_not_item, self.list_users, self.list_items, \
            self.dict_index = return_list_info(self.final_dataset)

        self.list_users = list(self.list_users)
        self.list_items = list(self.list_items)

        for train_set in self.individual_datasets:
            self.factors.append(self.simple_bpr(train_set))

    def simple_bpr(self, train_set):
        du, dni, lu, _, _ = return_list_info(train_set)
        lu = list(lu)
        p, q, bias = create_factors(len(self.list_users), len(self.list_items), num_factor)

        for _ in range(num_interactions):
            for z in range(num_interactions_baselines):
                u, i, j = sample_triple(du, dni, lu)
                u, i, j = self.dict_index['users'][u], self.dict_index['items'][i], self.dict_index['items'][j]
                rui = bias[i] + sum(np.array(p[u]) * np.array(q[i]))
                ruj = bias[j] + sum(np.array(p[u]) * np.array(q[j]))

                x_uij = rui - ruj

                try:
                    fun_exp = float(math.exp(-x_uij)) / float((1 + math.exp(-x_uij)))
                except OverflowError:
                    fun_exp = 0.5

                update_bias_i = fun_exp - reg_bias * bias[i]
                bias[i] += learn_rate * update_bias_i

                update_bias_j = fun_exp - reg_bias * bias[j]
                bias[j] += learn_rate * update_bias_j

                for num in range(num_factor):
                    w_uf = p[u][num]
                    h_if = q[i][num]
                    h_jf = q[j][num]

                    update_user = (h_if - h_jf) * fun_exp - reg_u * w_uf
                    p[u][num] = w_uf + learn_rate * update_user

                    update_item_i = w_uf * fun_exp - reg_i * h_if
                    q[i] = h_if + learn_rate * update_item_i

                    update_item_j = -w_uf * fun_exp - reg_j * h_jf
                    q[j] = h_jf + learn_rate * update_item_j

        return [p, q, bias]

    def train_weights(self):
        for _ in xrange(self.num_interactions):
            self.betas.append(np.random.uniform(0, 1, len(self.list_users)))

        for _ in xrange(num_interactions):
            for z in xrange(num_interactions_train_weights):
                u, i, j = sample_triple(self.dict_item, self.dict_not_item, self.list_users)
                u, i, j = self.dict_index['users'][u], self.dict_index['items'][i], self.dict_index['items'][j]

                suij = 0
                rui_list = list()
                ruj_list = list()

                for n, factor in enumerate(self.factors):
                    rui = factor[2][i] + sum(np.array(factor[0][u]) * np.array(factor[1][i]))
                    ruj = factor[2][j] + sum(np.array(factor[0][u]) * np.array(factor[1][j]))
                    rui_list.append(rui)
                    ruj_list.append(ruj)

                    suij += self.betas[n][u] * (rui-ruj)

                try:
                    fun_exp = float(math.exp(-suij)) / float((1 + math.exp(-suij)))
                except OverflowError:
                    fun_exp = 0.5

                for m, beta in enumerate(self.betas):
                    update_beta = (rui_list[m] - ruj_list[m]) * fun_exp - reg_bias_beta * self.betas[m][u]
                    self.betas[m][u] += learn_rate * update_beta

    def ensemble_ranks(self):
        for u, user in enumerate(self.list_users):
            list_items = list()
            for item in self.dict_not_item[user]:
                rui = 0
                for m in xrange(self.num_interactions):
                    try:
                        score = self.rankings[m][user].get(item, 0)
                        if score > 0:
                            score = (score - self.normalization[m][0]) / (
                                self.normalization[m][1] - self.normalization[m][0])
                            rui += self.betas[m][u] * score
                    except KeyError:
                        pass

                list_items.append([item, rui])

            list_items = sorted(list_items, key=lambda x: -x[1])
            self.final_ranking.append([user, list_items[:self.rank_number]])

    def write_ranking(self):
        write_ensemble = WriteFile(self.file_write, self.final_ranking, self.space_type)
        write_ensemble.write_prediction_file()
