import sys


class TreatData(object):
    def __init__(self, dataset_file, type_space):
        self.dataset_file = dataset_file
        self.type_space = type_space
        self.item_category_dict = dict()
        self.dataset_dict = dict()
        self.category_list = set()

    def read_dataset(self):
        with open(self.dataset_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.type_space)
                    user, item, feedback = inline[0], inline[1], inline[2]
                    self.dataset_dict.setdefault(user, {}).update({item: feedback})

    def read_attribute(self, category_file):
        with open(category_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.type_space)
                    inline[1] = inline[1].replace('\n', '')
                    self.category_list.add(inline[1])
                    self.item_category_dict.setdefault(inline[0], []).append(inline[1])
        print(self.item_category_dict['34'])
        print(self.category_list)

data_file = "C:\\Users\\Arthur\\OneDrive\\experiments2016\\datasets\\movielens\\dataset\\user_movies.dat"
attribute = "C:\\Users\\Arthur\\OneDrive\\experiments2016\\datasets\\movielens\\dataset\\movie_genres.dat"
# TreatData(data_file, '\t').read_dataset()

TreatData(data_file, '\t').read_attribute(attribute)
