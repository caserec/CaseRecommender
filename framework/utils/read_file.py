# coding=utf-8
"""
© 2016. Case Recommender All Rights Reserved (License GPL3)

This file is responsible for read external files.

The accepted data format is:
    user\titem\tinformation\n

* user and item must be integer!
* To change the spacing in the file, change the space_type var (default = \t).

Methods:
    - main_information and main_information_item_recommendation: returns a set of information about the dataset:
            * list with all users
            * list with all items
            * number of interactions
            * users interactions (dictionary with seen items and feedback for each user |
                                  number of interaction for each user)
            * items interactions (dictionary with users and feedback for each item |
                                  number of interaction for each item)

    - cross_fold_validation: return triples [user, item, feedback] and number of interactions

    - split_dataset: return triples [user, item, feedback], number of interactions and
                       users interactions (dictionary with seen items and feedback for each user |
                                  number of interaction for each user) for each feedback type.

    - rating_prediction: returns a set of specifics attributes from dataset in a dictionary:
            * dictionary with all interactions
            * list with all users
            * list with all items
            * dictionary of all users interaction
            * dictionary of all items interaction
            * mean of rates
    - read_rankings: return a dictionary and a list about one ranking
    - read_matrix: returns a data matrix
    - return_matrix: returns a dictionary with a matrix and other information

"""

import sys
import numpy as np
from framework.utils.extra_functions import check_error_file

__author__ = 'Arthur Fortes'


class ReadFile(object):
    def __init__(self, file_read, space_type='\t'):
        self.file_read = file_read
        self.space_type = space_type
        self.list_users = set()
        self.list_items = set()
        self.number_interactions = 0
        self.dict_users = dict()
        self.dict_items = dict()
        self.num_user_interactions = dict()
        self.num_items_interactions = dict()
        self.triple_dataset = list()
        self.individual_interaction = list()
        self.average_scores = dict()
        self.mean_feedback = 0

    def main_information(self):
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)

                    try:
                        user, item, feedback = int(inline[0]), int(inline[1]), inline[2]
                    except ValueError:
                        print('Error: Space type is invalid!')
                        sys.exit()

                    self.num_user_interactions[user] = self.num_user_interactions.get(user, 0) + 1
                    self.num_items_interactions[item] = self.num_items_interactions.get(item, 0) + 1
                    self.dict_users.setdefault(user, {}).update({item: feedback})
                    self.dict_items.setdefault(item, {}).update({user: feedback})
                    self.list_users.add(user)
                    self.list_items.add(item)
                    self.number_interactions += 1

        self.list_users = sorted(self.list_users)
        self.list_items = sorted(self.list_items)

    def main_information_item_recommendation(self):
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    try:
                        user, item = int(inline[0]), int(inline[1])
                    except ValueError:
                        print('Error: Space type is invalid!')
                        sys.exit()

                    self.num_user_interactions[user] = self.num_user_interactions.get(user, 0) + 1
                    self.num_items_interactions[item] = self.num_items_interactions.get(item, 0) + 1
                    self.list_users.add(user)
                    self.list_items.add(item)
                    self.dict_users.setdefault(user, []).append(item)

        self.list_users = sorted(self.list_users)
        self.list_items = sorted(self.list_items)

    def triple_information(self):
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), inline[2].replace("\n", "")
                    self.triple_dataset.append([user, item, feedback])

    def split_dataset(self):
        for i, feedback in enumerate(self.file_read):
            self.dict_users = dict()
            check_error_file(feedback)
            with open(feedback) as infile:
                for line in infile:
                    if line.strip():
                        inline = line.split(self.space_type)
                        self.number_interactions += 1
                        user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                        self.triple_dataset.append((user, item))
                        self.dict_users.setdefault(user, {}).update({item: feedback})
            self.individual_interaction.append(self.dict_users)

    def read_rankings(self):
        list_feedback = list()
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.dict_users.setdefault(user, {}).update({item: feedback})
                    list_feedback.append(feedback)
                    self.average_scores[user] = self.average_scores.get(user, 0) + feedback
                    self.num_user_interactions[user] = self.num_user_interactions.get(user, 0) + 1
        return self.dict_users, list_feedback

    def rating_prediction(self):
        dict_file = dict()
        d_feedback = dict()
        list_feedback = list()
        check_error_file(self.file_read)

        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    d_feedback.setdefault(user, {}).update({item: feedback})
                    self.triple_dataset.append((user, item, feedback))
                    self.dict_users.setdefault(user, set()).add(item)
                    self.dict_items.setdefault(item, set()).add(user)
                    self.list_users.add(user)
                    self.list_items.add(item)
                    self.mean_feedback += feedback
                    list_feedback.append(feedback)

        self.triple_dataset = sorted(self.triple_dataset)
        self.mean_feedback /= float(self.number_interactions)
        self.list_users = sorted(list(self.list_users))
        self.list_items = sorted(list(self.list_items))
        dict_file.update({'feedback': d_feedback, 'users': self.list_users, 'items': self.list_items,
                          'du': self.dict_users, 'di': self.dict_items, 'mean_rates': self.mean_feedback,
                          'list_feedback': self.triple_dataset, 'ni': self.number_interactions,
                          'max': max(list_feedback), 'min': min(list_feedback)})

        return dict_file

    def read_matrix(self):
        matrix = list()
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    inline = np.array(inline)
                    inline = np.delete(inline, len(inline)-1)
                    matrix.append(inline.astype(float))
        return np.array(matrix)

    def ensemble(self):
        dict_info = dict()

        for r, rank_file in enumerate(self.file_read):
            self.list_users = set()
            self.dict_users = dict()
            with open(rank_file) as infile:
                for line in infile:
                    if line.strip():
                        inline = line.split(self.space_type)
                        user, item, score = int(inline[0]), int(inline[1]), float(inline[2])
                        self.list_users.add(user)
                        self.dict_users.setdefault(user, list()).append([user, item, score])

            self.list_users = sorted(self.list_users)
            for user in self.list_users:
                n_rank = len(self.dict_users[user])
                for i, triple in enumerate(self.dict_users[user]):
                    self.dict_users[user][i][2] = n_rank - i

            dict_info.setdefault(r, dict()).update({"rank": self.dict_users, "users": self.list_users})

        return dict_info

    def ensemble_test(self):
        user_info = dict()
        rank_info = dict()
        for r, rank_file in enumerate(self.file_read):
            r_dict = dict()
            with open(rank_file) as infile:
                for line in infile:
                    if line.strip():
                        inline = line.split(self.space_type)
                        user, item, score = int(inline[0]), int(inline[1]), float(inline[2])
                        self.number_interactions += 1
                        self.list_users.add(user)
                        self.list_items.add(item)
                        self.dict_users.setdefault(user, {}).update({item: score})
                        r_dict.setdefault(user, {}).update({item: score})
            rank_info[r] = r_dict

        self.list_users = sorted(self.list_users)
        self.list_items = sorted(self.list_items)
        dict_non_seen = dict()

        for user in self.dict_users:
            dict_non_seen[user] = list(set(self.list_items) - set(self.dict_users[user]))
            user_info[user] = {"j": dict_non_seen[user], "i": self.dict_users[user].keys()}

        return user_info, self.list_users, self.list_items, self.number_interactions, rank_info

    def return_matrix(self, implicit=False):
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.list_users.add(user)
                    self.list_items.add(item)
                    self.dict_users.setdefault(user, {}).update({item: feedback})

        map_user = dict()
        map_index_user = dict()
        self.list_users = sorted(list(self.list_users))

        for u, user in enumerate(self.list_users):
            map_user[user] = u
            map_index_user[u] = user

        map_item = dict()
        map_index_item = dict()
        self.list_items = sorted(list(self.list_items))

        for i, item in enumerate(self.list_items):
            map_item[item] = i
            map_index_item[i] = item

        matrix = np.zeros((len(self.list_users), len(self.list_items)))

        for user in self.list_users:
            for item in self.dict_users[user]:
                if implicit:
                    matrix[map_user[user]][map_item[item]] = 1
                else:
                    matrix[map_user[user]][map_item[item]] = self.dict_users[user][item]
                self.dict_items.setdefault(map_item[item], set()).add(map_user[user])

        return {"matrix": matrix, "map_user": map_index_user, "map_item": map_index_item,
                "number_interactions": self.number_interactions, "di": self.dict_items, "mu": map_user,
                "users": self.list_users, "items": self.list_items, "feedback": self.dict_users}

    def read_metadata(self, l_items):
        dict_file = dict()
        d_feedback = dict()
        list_feedback = list()
        map_user = dict()
        map_index_user = dict()
        map_item = dict()
        map_index_item = dict()
        check_error_file(self.file_read)

        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    d_feedback.setdefault(user, {}).update({item: feedback})
                    self.triple_dataset.append((user, item, feedback))
                    self.dict_users.setdefault(user, set()).add(item)
                    self.dict_items.setdefault(item, set()).add(user)
                    self.list_items.add(item)
                    self.mean_feedback += feedback
                    list_feedback.append(feedback)

        self.triple_dataset = sorted(self.triple_dataset)
        self.mean_feedback /= float(self.number_interactions)
        self.list_users = sorted(list(l_items))
        self.list_items = sorted(list(self.list_items))

        for u, user in enumerate(self.list_users):
            map_user[user] = u
            map_index_user[u] = user

        for i, item in enumerate(self.list_items):
            map_item[item] = i
            map_index_item[i] = item

        matrix = np.zeros((len(self.list_users), len(self.list_items)))

        for user in self.list_users:
            try:
                for item in d_feedback[user]:
                    matrix[map_user[user]][map_item[item]] = d_feedback[user][item]
            except KeyError:
                pass

        dict_file.update({'feedback': d_feedback, 'items': self.list_users, 'metadata': self.list_items,
                          'di': self.dict_users, 'dm': self.dict_items, 'mean_rates': self.mean_feedback,
                          'list_feedback': self.triple_dataset, 'ni': self.number_interactions,
                          'max': max(list_feedback), 'min': min(list_feedback), 'matrix': matrix})

        return dict_file

    def return_bprmf(self):
        check_error_file(self.file_read)
        not_seen = dict()
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.list_users.add(user)
                    self.list_items.add(item)
                    self.dict_users.setdefault(user, list()).append(item)

        self.list_users = sorted(list(self.list_users))
        self.list_items = sorted(list(self.list_items))

        for user in self.list_users:
            not_seen[user] = list(set(self.list_items) - set(self.dict_users[user]))

        return {"users": self.list_users, "items": self.list_items, "feedback": self.dict_users, "not_seen": not_seen,
                "number_interactions": self.number_interactions}
