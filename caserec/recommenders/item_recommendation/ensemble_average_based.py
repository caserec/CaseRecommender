# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

Ensemble Average Based
    Literature:
        Improving Personalized Ranking in Recommender Systems with Multimodal Interactions.
        WIC 2014.
        http://dx.doi.org/10.1109/wi-iat.2014.34

Parameters
-----------

"""
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from caserec.utils.read_file import ReadFile
from caserec.utils.write_file import WriteFile
from caserec.utils.extra_functions import check_len_lists

__author__ = 'Arthur Fortes'


# utils
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


class EnsembleAverageBased(object):
    def __init__(self, list_train_files, list_rank_files, file_write, test_file=None, rank_number=10, space_type='\t'):
        self.list_train_files = list_train_files
        self.list_rank_files = list_rank_files
        self.file_write = file_write
        self.test_file = test_file
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
        self.averages = list()
        self.num_user_int = list()

    def read_ranking_files(self):
        for ranking_file in self.list_rank_files:
            ranking = ReadFile(ranking_file, space_type=self.space_type)
            rank_interaction, list_interaction = ranking.read_rankings()
            self.rankings.append(rank_interaction)
            self.normalization.append([min(list_interaction), max(list_interaction)])
            self.averages.append(ranking.average_scores)
            self.num_user_int.append(ranking.num_user_interactions)

    def treat_interactions(self):
        for num, interaction_file in enumerate(self.list_train_files):
            interaction = ReadFile(interaction_file, space_type=self.space_type)
            interaction.triple_information()
            self.individual_datasets.append(interaction.triple_dataset)
            self.final_dataset += interaction.triple_dataset

        self.dict_item, self.dict_not_item, self.list_users, self.list_items, \
            self.dict_index = return_list_info(self.final_dataset)

        self.list_users = list(self.list_users)
        self.list_items = list(self.list_items)

    def ensemble_ranks(self):
        for u, user in enumerate(self.list_users):
            list_items = list()
            for item in self.dict_not_item[user]:
                scores = list()
                control = 0
                for m in range(self.num_interactions):
                    try:
                        score = self.rankings[m][user].get(item, 0)
                        if score > 0:
                            average = float(self.averages[m][user]) / float(self.num_user_int[m][user])
                            if score >= average:
                                control += 1

                            score = (score - self.normalization[m][0]) / (
                                self.normalization[m][1] - self.normalization[m][0])
                            scores.append(score)

                    except KeyError:
                        pass

                if len(scores) == self.num_interactions:

                    if control == self.num_interactions:
                        rui = (float(sum(scores)) / float(len(scores)))
                    else:
                        rui = max(scores)
                    list_items.append([item, rui])

            list_items = sorted(list_items, key=lambda x: -x[1])
            self.final_ranking.append([user, list_items[:self.rank_number]])

    def write_ranking(self):
        self.final_ranking = sorted(self.final_ranking, key=lambda x: x[0])
        write_ensemble = WriteFile(self.file_write, self.final_ranking, self.space_type)
        write_ensemble.write_recommendation()

    def evaluate(self, measures):
        res = ItemRecommendationEvaluation().evaluation_ranking(self.final_ranking, self.test_file)
        evaluation = 'Eval:: '
        for measure in measures:
            evaluation += measure + ': ' + str(res[measure]) + ' '
        print(evaluation)

    def execute(self, measures=('Prec@5', 'Prec@10', 'NDCG@5', 'NDCG@10', 'MAP@5', 'MAP@10')):
        # methods
        print("[Case Recommender: Item Recommendation > Ensemble Average Based Algorithm]\n")
        # call internal methods
        self.read_ranking_files()
        print("Read Ranking Files...")
        self.treat_interactions()
        self.ensemble_ranks()
        print("Finished Ensemble interactions...")
        self.write_ranking()

        if self.test_file is not None:
            self.evaluate(measures)
