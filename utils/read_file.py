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

    def main_information(self):
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), inline[2]
                    self.num_user_interactions[user] = self.num_user_interactions.get(user, 0) + 1
                    self.num_items_interactions[item] = self.num_items_interactions.get(item, 0) + 1
                    self.list_users.add(user)
                    self.list_items.add(item)

                    if user in self.user_interactions:
                        self.user_interactions[user].update({item: feedback})
                    else:
                        self.user_interactions[user] = {item: feedback}

                    if item in self.item_interactions:
                        self.item_interactions[item].update({user: feedback})
                    else:
                        self.item_interactions[item] = {user: feedback}

        self.list_users = sorted(self.list_users)
        self.list_items = sorted(self.list_items)

    def cross_fold_validation(self):
        with open(self.file_read) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.space_type)
                    self.number_interactions += 1
                    user, item, feedback = int(inline[0]), int(inline[1]), inline[2]
                    self.triple_dataset.append([user, item, feedback])
