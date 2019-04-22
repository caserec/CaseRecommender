"""

    CoRec and ECoRec: Co-training Algorithm
    [Rating Prediction]

    Literature:
        DA COSTA, ARTHUR F.; MANZATO, MARCELO G. ; CAMPELLO, RICARDO J.G.B.
        CoRec: https://dl.acm.org/citation.cfm?doid=3167132.3167209
        ECoRec: https://doi.org/10.1016/j.eswa.2018.08.020

    PS.: 
    
    1. I'm still working in this implementation and this algorithm is not the same of the article. 
    For this reason, this algoritm is not in pypi version.
    2. Soon as possible, the new version will be uploaded.
    3. This algorithm was implemented with parallel processing.

"""

# © 2018. Case Recommender (MIT License)

from multiprocessing.pool import Pool
from random import shuffle
import numpy as np
import os

from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.svdplusplus import SVDPlusPlus
from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.utils.process_data import ReadFile
from caserec.utils.extra_functions import ComputeBui

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class ECoRec(object):
    def __init__(self, train_file, test_file, recommenders=(1, 2), confidence_measure='vi', number_sample=10, m=None,
                 sep='\t', ensemble_method=False):

        """

        (E)CoRec for rating prediction

        This algorithm is based on a co-training approach, named (E)CoRec, that drives two or more recommenders to
        agree with each others’ predictions to generate their own. The output of this algorithm is n (where n is the
        number of used recommenders) new enriched user x item matrix, which can be used as new training sets.

        Usage::

            >> ECoRec(tr, te, number_sample=10, confidence_measure='su').compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param recommenders: Recommenders in which should be used as regressors to predict unlabeled sets. At least 2
        should be selected. Options:
            - 1: Item KNN
            - 2: User KNN
            - 3: Matrix Factorization
            - 4: SVD++
        :type recommenders: tuple, default (1,2)

        :param confidence_measure: Confidence Measure to calculate the precision of the predicted sample. Options:
               - 'pc': proposed metric in the articles
               - 'vi': Variability for item
               - 'su': supported for users
               - 'si': supported for items
        :type confidence_measure: str, default 'vi'

        :param number_sample: Number of new samples (unlabeled samples) per user, which should be labeled
        :type number_sample: int, default 10

        :param m: Number of most confident examples to select in each interaction
        :type m: int, default None

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param ensemble_method: Active ensemble method (ECoRec), combining the n other recommenders outputs
        :type ensemble_method: bool, default False

        """

        self.train_file = train_file
        self.test_file = test_file
        self.train_set = ReadFile(train_file, sep=sep).read()
        self.test_set = ReadFile(test_file, sep=sep).read()
        self.recommenders = recommenders
        self.confidence_measure = confidence_measure
        self.number_sample = number_sample
        self.m = m
        self.sep = sep
        self.ensemble_method = ensemble_method

        # internal vars
        self.number_of_recommenders = len(recommenders)
        self.unlabeled_set = None
        self.unlabeled_data = dict()
        self.labeled_files = dict()
        self.unlabeled_files = dict()
        self.ensemble_file = None

        self.empty = False
        self.recommenders_predictions = None
        self.recommenders_confident = None

        self.rec_conf = dict()
        self.l_test = list()

        np.random.seed(123)

    def create_unlabeled_set(self):
        """
        Create a pool U' for unlabeled set U (Create unlabeled_1.dat, unlabeled_2.dat ... unlabeled_N.dat)
        """
        unlabeled_data = list()

        for user in self.train_set['users']:
            sample = list(set(self.train_set['items']) - set(self.train_set['feedback'].get(user, [])))
            sub_sample = list()
            for item in sample:
                sub_sample.append((user, item))
            np.random.shuffle(sub_sample)

            unlabeled_data += sub_sample[:self.number_sample]

        # Calculate the number of M confident examples based on unlabeled sample
        if self.m is None:
            self.m = int(len(unlabeled_data) * .1)

        unlabeled_data = sorted(unlabeled_data, key=lambda x: (x[0], x[1]))
        self.unlabeled_set = unlabeled_data.copy()

    def create_initial_files(self):
        """
        Create labeled and unlabeled files for N recommenders
        """
        for r in self.recommenders:
            labeled_file = os.path.dirname(self.train_file) + '/labeled_set_' + str(r) + '.dat'
            self.labeled_files.setdefault(r, labeled_file)
            unlabeled_file = os.path.dirname(self.train_file) + '/unlabeled_set_' + str(r) + '.dat'
            self.unlabeled_files.setdefault(r, unlabeled_file)

            self.write_with_dict(labeled_file, self.train_set['feedback'])
            self.write_file(self.unlabeled_set, unlabeled_file, score=False)
            self.unlabeled_data.setdefault(r, self.unlabeled_set)

        if self.ensemble_method:
            self.ensemble_file = os.path.dirname(self.train_file) + '/ensemble_set.dat'
            self.write_with_dict(self.ensemble_file, self.train_set['feedback'])

    def train_model(self):
        """
        Train the model in a co-training process
        """

        epoch = 0
        while not self.empty:
            print("Epoch:: ", epoch)
            self.recommenders_predictions = dict()
            self.recommenders_confident = dict()

            if not self.run_recommenders_parallel():
                break

            self.learn_confident_parallel()
            self.update_data()

            epoch += 1

    # Proposed Confidence
    def pc(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        bui = ComputeBui(label_data).execute()

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]

            nu = len(label_data['items_seen_by_user'].get(user, []))
            ni = len(label_data['users_viewed_item'].get(item, []))

            # compute bui and error
            try:
                den = np.fabs(bui[user][item] - feedback)
            except KeyError:
                den = np.fabs(label_data['mean_value'] - feedback)

            if den == 0:
                den = 0.001

            # compute confidence
            c = (nu * ni) * (1 / den)

            if c != 0:
                confident.append((user, item, feedback, c))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # Variability for item
    def vi(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        ci = {}

        for item in label_data['items']:
            list_rating = []
            for user in label_data['users_viewed_item'][item]:
                list_rating.append(label_data['feedback'][user][item])
            ci[item] = np.std(list_rating)

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            confident.append((user, item, feedback, ci.get(item, 0)))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # Support for user
    def su(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        ci = {}

        for user in label_data['users']:
            ci[user] = len(label_data['items_seen_by_user'][user])

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            confident.append((user, item, feedback, ci.get(user, 0)))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # Support for item
    def si(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        ci = {}

        for item in label_data['items']:
            ci[item] = len(label_data['users_viewed_item'][item])

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            confident.append((user, item, feedback, ci.get(item, 0)))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    def learn_confident_parallel(self):
        pool = Pool()

        method = getattr(self, self.confidence_measure)
        result = pool.map(method, self.recommenders)
        pool.close()
        pool.join()

        for n, r in enumerate(self.recommenders):
            self.rec_conf.setdefault(r, result[n][0])
            self.recommenders_confident.setdefault(r, result[n][1])

    def update_data(self):
        np.random.seed(0)
        n_rec = list(self.recommenders).copy()
        cond = True
        while cond:
            shuffle(n_rec)
            if not [i for i, j in zip(n_rec, self.recommenders) if i == j]:
                cond = False

        for n, r in enumerate(self.recommenders):
            if self.recommenders_confident[r]:
                self.update_file(self.recommenders_confident[n_rec[n]], self.labeled_files[r])

                rec_conf = [(x[0], x[1]) for x in self.recommenders_confident[r]]
                self.unlabeled_data[r] = list(set(self.unlabeled_data[r]) - set(rec_conf))
                self.write_file(self.unlabeled_data[r], self.unlabeled_files[r], score=False)
            else:
                self.empty = True

    def ensemble(self):
        ensemble_data = list()
        final_rui = {}
        final_dev = {}

        for r in self.recommenders:
            for conf_sample in self.rec_conf[r]:
                user, item, rui, conf = conf_sample
                if final_rui.get(user, -1) == -1:
                    final_rui.setdefault(user, {}).update({item: rui})
                    final_dev.setdefault(user, {}).update({item: conf})
                else:
                    if final_rui[user].get(item, -1) == -1:
                        final_rui[user][item] = (final_rui[user].get(item, 0) + rui)
                        final_dev[user][item] = (final_dev[user].get(item, 0) + conf)
                    else:
                        if conf > final_dev[user][item]:
                            final_dev[user][item] = rui

        for user in final_rui:
            for item in final_rui[user]:
                rui = final_rui[user][item]

                if rui > self.train_set['max_value']:
                    rui = self.train_set['max_value']
                elif rui < self.train_set['min_value']:
                    rui = self.train_set['min_value']

                ensemble_data.append((user, item, rui))

        self.update_file(ensemble_data, self.ensemble_file)

    def write_file(self, triples, write_file, score=True):
        with open(write_file, 'w') as infile:
            if score:
                for t in triples:
                    infile.write(str(t[0]) + self.sep + str(t[1]) + self.sep + str(t[2]) + '\n')
            else:
                for t in triples:
                    infile.write(str(t[0]) + self.sep + str(t[1]) + self.sep + '1.0 \n')

    def write_with_dict(self, write_file, dict_data):
        """
        Method to write using data as dictionary. e.g.: user: {item : score}

        """

        with open(write_file, 'w') as infile:
            for user in dict_data:
                for pair in dict_data[user]:
                    infile.write('%d%s%d%s%f\n' % (user, self.sep, pair, self.sep,
                                                   dict_data[user][pair]))

    def update_file(self, triples, write_file):
        with open(write_file, 'a') as infile:
            for t in triples:
                infile.write(str(t[0]) + self.sep + str(t[1]) + self.sep + str(t[2]) + '\n')

    def run_recommenders_parallel(self):
        """
            create a method to run in parallel the recommenders during the co-training process

        :return: True ou False, if threshold is reached (False) or not (True)
        """

        flag = True

        pool = Pool()
        result = pool.map(self.run_recommenders, self.recommenders)
        pool.close()
        pool.join()

        for n, r in enumerate(self.recommenders):
            if not result[n][1]:
                flag = False
            else:
                self.recommenders_predictions.setdefault(r, result[n][0])

        return flag

    def run_recommenders(self, r):
        """
        1: Item KNN
        2: User KNN
        3: Matrix Factorization
        4: SVD++
        """

        flag = True

        if not self.unlabeled_data[r]:
            flag = False

            return [], flag

        else:
            if r == 1:
                rec = ItemKNN(self.labeled_files[r], self.unlabeled_files[r], as_similar_first=True)
                rec.read_files()
                rec.init_model()
                rec.train_baselines()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)
            elif r == 2:
                rec = UserKNN(self.labeled_files[r], self.unlabeled_files[r])
                rec.read_files()
                rec.init_model()
                rec.train_baselines()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)
            elif r == 3:
                rec = MatrixFactorization(self.labeled_files[r], self.unlabeled_files[r], random_seed=1, baseline=True)
                rec.read_files()
                rec.init_model()
                rec.fit()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)
            elif r == 4:
                rec = SVDPlusPlus(self.labeled_files[r], self.unlabeled_files[r], random_seed=1)
                rec.read_files()
                rec.fit()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)

            else:
                raise NameError('Invalid Recommender!')

            return rec.predictions, flag

    @staticmethod
    def transform_dict(list_to_transform):
        new_dict = dict()
        for t in list_to_transform:
            user, item, feedback = t[0], t[1], t[2]
            new_dict.setdefault(user, {}).update({item: feedback})
        return new_dict

    def del_unlabeled_files(self):
        for f in self.unlabeled_files:
            os.remove(self.unlabeled_files[f])

    def compute(self):
        self.create_unlabeled_set()
        self.create_initial_files()

        self.train_model()

        if self.ensemble_method:
            self.ensemble()

        self.del_unlabeled_files()
