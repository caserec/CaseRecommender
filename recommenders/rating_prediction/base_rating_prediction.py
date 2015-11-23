from utils.read_file import ReadFile

__author__ = 'Arthur Fortes'


class BaseRatingPrediction(object):
    def __init__(self, train_file, test_file='', space_type='\t'):
        self.space_type = space_type
        self.matrix = []
        self.mean_feedback = 0
        self.num_interactions = 0
        self.test = ''
        self.regBi = 10
        self.regBu = 15
        self.bu = dict()
        self.bi = dict()
        self.bui = dict()
        self.dict_item_id = dict()
        self.dict_user_id = dict()
        self.train = ReadFile(train_file, self.space_type)
        self.train.main_information()

        if test_file != '':
            self.test = ReadFile(test_file, self.space_type)
            self.test.main_information()
            self.dataset_users = sorted(list(set(self.train.list_users + self.test.list_users)))
            self.dataset_items = sorted(list(set(self.train.list_items + self.test.list_items)))

        else:
            self.dataset_users = self.train.list_users
            self.dataset_items = self.train.list_items

        for item_id, item in enumerate(self.dataset_items):
            self.dict_item_id[item] = item_id

        for user_id, user in enumerate(self.dataset_users):
            self.dict_user_id[user] = user_id

    def fill_user_item_matrix(self):
        self.matrix = [[0 for _ in range(len(self.dataset_items))] for _ in range(len(self.dataset_users))]
        for u, user in enumerate(self.dataset_users):
            for item in self.train.user_interactions[user]:
                self.matrix[u][self.dict_item_id[item]] = float(self.train.user_interactions[user][item])
                self.num_interactions += 1
                self.mean_feedback += float(self.train.user_interactions[user][item])
        self.mean_feedback /= float(self.num_interactions)

    def fill_item_user_matrix(self):
        self.matrix = [[0 for _ in range(len(self.dataset_users))] for _ in range(len(self.dataset_items))]
        for u, user in enumerate(self.dataset_users):
            for item in self.train.user_interactions[user]:
                self.matrix[self.dict_item_id[item]][u] = float(self.train.user_interactions[user][item])
                self.num_interactions += 1
                self.mean_feedback += float(self.train.user_interactions[user][item])
        self.mean_feedback /= float(self.num_interactions)

    def fill_attribute_matrix(self, attr_file):
        pass

    def train_baselines(self):
        print('Training baselines...')
        for i in xrange(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def compute_bi(self):
        # bi = (rui - mi - bu) / (regBi + number of interactions)
        self.bi = dict()

        for item in self.dataset_items:
            cont = 0
            try:
                for user in self.train.item_interactions[item]:
                    self.bi[item] = self.bi.get(item, 0) + float(self.train.item_interactions[item][user]) - \
                                         self.mean_feedback - self.bu.get(user, 0)
                    cont += 1
            except KeyError:
                pass

            if cont > 1:
                self.bi[item] = float(self.bi[item]) / float(self.regBi + cont)

    def compute_bu(self):
        # bu = (rui - mi - bi) / (regBu + number of interactions)
        self.bu = dict()
        for user in self.dataset_users:
            cont = 0
            for item in self.train.user_interactions[user]:
                self.bu[user] = self.bu.get(user, 0) + float(self.train.user_interactions[user][item]) - \
                                 self.mean_feedback - self.bi.get(item, 0)
                cont += 1
            if cont > 1:
                self.bu[user] = float(self.bu[user]) / float(self.regBu + cont)

    def compute_bui(self):
        # bui = mi + bu + bi
        for user in self.dataset_users:
            for item in self.dataset_items:
                if self.bui.get(user, []):
                    try:
                        self.bui[user].update({item: self.mean_feedback + self.bu[user] + self.bi[item]})
                    except KeyError:
                        pass
                else:
                    self.bui[user] = {item: self.mean_feedback + self.bu[user] + self.bi[item]}
        del self.bu
        del self.bi
