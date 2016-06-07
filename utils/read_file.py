import sys
import numpy as np
from utils.error_functions import check_error_file

__author__ = 'Arthur Fortes'

'''
This file is responsible for read external files.

The accepted data format is:

user\titem\tinformation\n

* user and item must be integer!
* To change the spacing in the file, change the space_type var (default = \t).

Methods:
    - [main_information] and [main_information_item_recommendation]: returns a set of information about the dataset:
            * list with all users
            * list with all items
            * number of interactions
            * users interactions (dictionary with seen items and feedback for each user |
                                  number of interaction for each user)
            * items interactions (dictionary with users and feedback for each item |
                                  number of interaction for each item)

    - [cross_fold_validation]: return triples [user, item, feedback] and number of interactions

    - [split_dataset]: return triples [user, item, feedback], number of interactions and
                       users interactions (dictionary with seen items and feedback for each user |
                                  number of interaction for each user) for each feedback type.

    - [rating_prediction]: returns a set of specifics attributes from dataset in a dictionary:
            * dictionary with all interactions
            * list with all users
            * list with all items
            * dictionary of all users interaction
            * dictionary of all items interaction
            * mean of rates

    - [read_matrix]: returns a data matrix

'''


class ReadFile(object):
    def __init__(self, file_read, space_type='\t'):
        self.file_read = file_read
        self.space_type = space_type
        self.list_users = set()
        self.list_items = set()
        self.number_interactions = 0
        self.user_interactions = dict()
        self.item_interactions = dict()
        self.num_user_interactions = dict()
        self.num_items_interactions = dict()
        self.triple_dataset = list()
        self.individual_interaction = list()
        self.average_scores = dict()

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
                    self.user_interactions.setdefault(user, {}).update({item: feedback})
                    self.item_interactions.setdefault(item, {}).update({user: feedback})
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
                    self.user_interactions.setdefault(user, []).append(item)

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
            self.user_interactions = dict()
            check_error_file(feedback)
            with open(feedback) as infile:
                for line in infile:
                    if line.strip():
                        inline = line.split(self.space_type)
                        self.number_interactions += 1
                        user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                        self.triple_dataset.append((user, item))
                        self.user_interactions.setdefault(user, {}).update({item: feedback})
            self.individual_interaction.append(self.user_interactions)

    def read_rankings(self):
        list_feedback = list()
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.user_interactions.setdefault(user, {}).update({item: feedback})
                    list_feedback.append(feedback)
                    self.average_scores[user] = self.average_scores.get(user, 0) + feedback
                    self.num_user_interactions[user] = self.num_user_interactions.get(user, 0) + 1
        return self.user_interactions, list_feedback

    def rating_prediction(self):
        dict_file = dict()
        d_feedback = dict()
        list_users = set()
        list_items = set()
        list_feedback = list()
        dict_items = dict()
        dict_users = dict()
        mean_rates = 0
        num_interactions = 0

        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split("\t")
                    num_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    d_feedback.setdefault(user, {}).update({item: feedback})
                    list_feedback.append((user, item, feedback))
                    dict_users.setdefault(user, set()).add(item)
                    dict_items.setdefault(item, set()).add(user)
                    list_users.add(user)
                    list_items.add(item)
                    mean_rates += feedback

        list_feedback = sorted(list_feedback)
        mean_rates /= float(num_interactions)
        list_users = sorted(list(list_users))
        list_items = sorted(list(list_items))
        dict_file.update({'feedback': d_feedback, 'users': list_users, 'items': list_items, 'du': dict_users,
                          'di': dict_items, 'mean_rates': mean_rates, 'list_feedback': list_feedback,
                          'ni': num_interactions})

        return dict_file

    def read_matrix(self):
        matrix = list()
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split("\t")
                    inline = np.array(inline)
                    matrix.append(inline.astype(float))
        return np.array(matrix)
