import numpy as np
from scipy.spatial.distance import squareform, pdist


class TreatData(object):
    def __init__(self, input_file, type_space="\t"):
        self.input_file = input_file
        self.type_space = type_space
        self.item_category_dict = dict()
        self.dataset_dict = dict()
        self.category_list = set()

    def read_dataset(self):
        with open(self.input_file) as infile:
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

    def compute_write_distance_matrix(self, distance, write_file, user_matrix=True):
        lu = set()
        li = set()

        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.type_space)
                    user, item, feedback = int(inline[0]), int(inline[1]), float(inline[2])
                    self.dataset_dict.setdefault(user, {}).update({item: feedback})
                    lu.add(user)
                    li.add(item)

        matrix = np.zeros((len(lu), len(li)))
        lu = sorted(list(lu))
        li = sorted(list(li))

        map_items = dict()
        map_users = dict()
        for i, item in enumerate(li):
            map_items.update({item: i})
        for u, user in enumerate(lu):
            map_users.update({user: u})

        for user in self.dataset_dict:
            for item in self.dataset_dict[user]:
                matrix[map_users[user]][map_items[item]] = self.dataset_dict[user][item]

        if not user_matrix:
            matrix = matrix.T

        matrix = np.float32(squareform(pdist(matrix, distance)))
        matrix = 1 - matrix
        with open(write_file, "w") as infile:
            for i in xrange(len(matrix)):
                for j in xrange(len(matrix[0])):
                    infile.write(str(matrix[i][j]) + self.type_space)
                infile.write("\n")
