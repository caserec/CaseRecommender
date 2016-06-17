"""

    test

"""
from random import shuffle, sample

import math

from recommenders.rating_prediction.itemknn import ItemKNN
from recommenders.rating_prediction.userknn import UserKNN


class MVLrec(object):
    def __init__(self, train_set, test_set, percent=0.2, recommender1="itemknn", recommender2="userknn",
                 times=20, k=1000):
        self.train = train_set
        self.test = test_set
        self.percent = percent
        self.times = times
        self.k_confident = k
        self.recommender1 = recommender1
        self.recommender2 = recommender2
        self.r1 = list()
        self.r2 = list()
        self.labeled_set = list()
        self.unlabeled_set = list()
        self.ensemble_labeled = list()
        self.predictions = dict()

        self.treat_train()
        self.rec_1 = {"labeled": self.labeled_set, "unlabeled": self.unlabeled_set}
        self.rec_2 = {"labeled": self.labeled_set, "unlabeled": self.unlabeled_set}
        self.train_recommender()
        self.ensemble()
        self.generate_traditional_recommenders()
        self.prediction()

    def treat_train(self):
        tuples = self.train['list_feedback']
        shuffle(tuples)
        self.unlabeled_set = sorted(tuples[:int(len(tuples) * self.percent)], key=lambda x: x[0])
        self.labeled_set = sorted(tuples[int(len(tuples) * self.percent):], key=lambda x: x[0])

    def train_recommender(self):
        for x in xrange(self.times):
            di1 = return_list_info(self.rec_1["labeled"])
            di2 = return_list_info(self.rec_2["labeled"])
            di_test1 = return_list_info(self.rec_1["unlabeled"])
            di_test2 = return_list_info(self.rec_2["unlabeled"])

            r1 = recommender(self.recommender1, di1, di_test1)
            r2 = recommender(self.recommender2, di2, di_test2)

            self.rec_1["labeled"], self.rec_2["unlabeled"] = self.update_labeled_set(self.rec_1["labeled"],
                                                                                     r2.predictions, di2)
            self.rec_2["labeled"], self.rec_1["unlabeled"] = self.update_labeled_set(self.rec_2["labeled"],
                                                                                     r1.predictions, di1)

            if self.rec_1["unlabeled"] == [] and self.rec_2["unlabeled"] == []:
                break
            else:
                print len(self.rec_1["unlabeled"]), len(self.rec_2["unlabeled"])

    def update_labeled_set(self, labeled, unlabeled, di):
        confidence_list = list()
        new_labeled = list()

        new_unlabeled = unlabeled
        sample(new_unlabeled, int(len(new_labeled)*0.7))
        lab = return_list_info(labeled)

        for p in new_unlabeled:
            user, item, feedback = p[0], p[1], p[2]
            error = (math.fabs(lab["mu"][user] - feedback) + math.fabs(lab["mi"][item] - feedback)) / 10.0
            confidence = (di["users"][user] * di["items"][item]) / float(di["count"]) - error
            confidence_list.append((user, item, feedback, confidence))
        confidence_list = sorted(confidence_list, key=lambda x: -x[3])

        for c in confidence_list[:self.k_confident]:
            new_labeled.append((c[0], c[1], c[2]))

        labeled += new_labeled
        unlabeled = list(set(unlabeled) - set(new_labeled))

        return labeled, unlabeled

    def ensemble(self):
        di1 = return_list_info(self.rec_1["labeled"])
        di2 = return_list_info(self.rec_2["labeled"])

        list_feedback = di1["list_feedback"] + di2["list_feedback"]
        list_feedback = list(set(tuple(i) for i in list_feedback))

        for pair in list_feedback:
            user, item = pair[0], pair[1]
            c1 = (di1["users"][user] * di1["items"][item]) / float(di1["count"])
            c2 = (di2["users"][user] * di2["items"][item]) / float(di2["count"])
            cont = 2

            try:
                n1 = di1["feedback"][user][item]
            except KeyError:
                n1 = 0
                cont -= 1
            try:
                n2 = di2["feedback"][user][item]
            except KeyError:
                n2 = 0
                cont -= 1

            try:
                score = (c1 / float((c1 + c2)) * n1) + (c2 / float(c1 + c2) * n2)
            except ZeroDivisionError:
                score = (n1 + n2) / float(cont)

            self.ensemble_labeled.append((user, item, score))

    def generate_traditional_recommenders(self):
        labeled = return_list_info(self.labeled_set)
        unlabeled = return_list_info(self.unlabeled_set)

        r1 = recommender(self.recommender1, labeled, unlabeled)
        r2 = recommender(self.recommender2, labeled, unlabeled)

        self.r1 = self.labeled_set + r1.predictions
        self.r2 = self.labeled_set + r2.predictions

    def prediction(self):
        # recommender 1 - >
        di1 = return_list_info(self.rec_1["labeled"])
        r1 = recommender(self.recommender1, di1, self.test)
        # recommender 2 - >
        di2 = return_list_info(self.rec_2["labeled"])
        r2 = recommender(self.recommender2, di2, self.test)
        # ensemble ->
        die = return_list_info(self.ensemble_labeled)
        re1 = recommender(self.recommender1, die, self.test)
        re2 = recommender(self.recommender2, die, self.test)

        # traditional approach
        #    recommender 1 - >
        di1 = return_list_info(self.r1)
        rt1 = recommender(self.recommender1, di1, self.test)
        #    recommender 2 - >
        di2 = return_list_info(self.r2)
        rt2 = recommender(self.recommender2, di2, self.test)

        self.predictions = {"MVLrec": [("Recommender 1", r1.predictions), ("Recommender 2", r2.predictions),
                            ("Recommender 1 with Ensemble", re1.predictions),
                            ("Recommender 2 with ensemble", re2.predictions)],
                            "Traditional Approaches": [("Recommender 1", rt1.predictions),
                                                       ("Recommender 2", rt2.predictions)]}


def recommender(type_recommender, train_set, test_set):
    if type_recommender == "itemknn":
        predictions = ItemKNN(train_set, test_set)
        return predictions
    elif type_recommender == "userknn":
        predictions = UserKNN(train_set, test_set)
        return predictions


def return_list_info(list_info):
    dict_info = {"users": dict(), "items": dict(), "count": 0, "feedback": dict(), "du": dict(), "di": dict(),
                 "mean_rates": 0, "list_feedback": list(), "mu": dict(), "mi": dict()}
    lu = set()
    li = set()

    for triple in list_info:
        user, item, feedback = triple[0], triple[1], triple[2]
        li.add(item)
        lu.add(user)
        dict_info["list_feedback"].append([user, item])
        dict_info["count"] += 1
        dict_info["users"][user] = dict_info['users'].get(user, 0) + 1
        dict_info["items"][item] = dict_info['items'].get(item, 0) + 1
        dict_info["mu"][user] = dict_info['mu'].get(user, 0) + feedback
        dict_info["mi"][item] = dict_info['mi'].get(item, 0) + feedback
        dict_info["mean_rates"] += feedback
        dict_info["feedback"].setdefault(user, {}).update({item: feedback})
        dict_info["du"].setdefault(user, set()).add(item)
        dict_info["di"].setdefault(item, set()).add(user)

    dict_info["mean_rates"] /= dict_info["count"]
    for u in lu:
        dict_info["mu"][u] /= dict_info["users"][u]
    for i in li:
        dict_info["mi"][i] /= dict_info["items"][i]
    return dict_info
