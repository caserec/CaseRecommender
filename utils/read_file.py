import sys
from utils.error_functions import check_error_file

__author__ = 'Arthur Fortes'

'''
This file is responsible for read external files.

The accepted data format is:

user\titem\tinformation\n

* user and item must be integer!
* To change the spacing in the file, change the space_type var (default = \t).

Methods:
    - [main_information]: returns a set of information about the dataset:
            * list with all users
            * list with all items
            * number of interactions
            * users interactions (dictionary with seen items and feedback for each user |
                                  number of interaction for each user)
            * items interactions (dictionary with users and feedback for each item |
                                  number of interaction for each item)

    - [cross_fold_validation]: return triples [user, item, feedback] and number of interactions

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

    def cross_fold_validation(self):
        check_error_file(self.file_read)
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), inline[2]
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
