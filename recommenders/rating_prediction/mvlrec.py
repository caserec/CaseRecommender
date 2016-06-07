"""

    test

"""
from random import shuffle
from recommenders.rating_prediction.itemknn import ItemKNN
from recommenders.rating_prediction.userknn import UserKNN


class MVLrec(object):
    def __init__(self, train_set, percent=0.2, recommender1="itemknn", recommender2="userknn", times=20, k=1000):
        print 'entrou'
        self.train = train_set
        self.percent = percent
        self.times = times
        self.k_confident = k
        self.recommender1 = recommender1
        self.recommender2 = recommender2
        self.labeled_set = list()
        self.unlabeled_set = list()
        self.predictions = dict()

        self.treat_train()
        self.train_recommender()

    def treat_train(self):
        print 'tratou entrada'
        tuples = self.train['list_feedback']
        shuffle(tuples)
        self.unlabeled_set = sorted(tuples[:int(len(tuples) * self.percent)], key=lambda x: x[0])
        self.labeled_set = sorted(tuples[int(len(tuples) * self.percent):], key=lambda x: x[0])

    def train_recommender(self):
        print 'vai trienar'
        rec_1 = {"labeled": self.labeled_set, "unlabeled": self.unlabeled_set}
        rec_2 = {"labeled": self.labeled_set, "unlabeled": self.unlabeled_set}

        for x in xrange(self.times):
            print 'interacao: ', x
            di1 = return_list_info(rec_1["labeled"])
            di2 = return_list_info(rec_2["labeled"])
            di_test1 = return_list_info(rec_1["unlabeled"])
            di_test2 = return_list_info(rec_2["unlabeled"])
            r1 = recommender(self.recommender1, di1, di_test1)
            print 'treinou rec1'
            r2 = recommender(self.recommender2, di2, di_test2)
            print 'treinou rec2'
            rec_1["labeled"], rec_2["unlabeled"] = self.update_labeled_set(rec_1["labeled"], r2.predictions, di2)
            rec_2["labeled"], rec_1["unlabeled"] = self.update_labeled_set(rec_2["labeled"], r1.predictions, di1)
            print len(rec_1["unlabeled"])
            if rec_1["unlabeled"] == [] and rec_2["unlabeled"] == []:

                break

    def update_labeled_set(self, labeled, unlabeled, di):
        confidence_list = list()
        new_labeled = list()

        for p in unlabeled:
            user, item, feedback = p[0], p[1], p[2]
            confidence = (di["users"][user] * di["items"][item]) / float(di["count"])
            confidence_list.append((user, item, feedback, confidence))
        confidence_list = sorted(confidence_list, key=lambda x: -x[3])

        for c in confidence_list[:self.k_confident]:
            new_labeled.append((c[0], c[1], c[2]))

        labeled += new_labeled
        unlabeled = list(set(unlabeled) - set(new_labeled))

        return labeled, unlabeled


def recommender(type_recommender, train_set, test_set):
    if type_recommender == "itemknn":
        predictions = ItemKNN(train_set, test_set)
        return predictions
    elif type_recommender == "userknn":
        predictions = UserKNN(train_set, test_set)
        return predictions


def return_list_info(list_info):
    dict_info = {"users": dict(), "items": dict(), "count": 0, "feedback": dict(), "du": dict(), "di": dict(),
                 "mean_rates": 0}

    for triple in list_info:
        user, item, feedback = triple[0], triple[1], triple[2]
        dict_info["count"] += 1
        dict_info["users"][user] = dict_info['users'].get(user, 0) + 1
        dict_info["items"][item] = dict_info['items'].get(item, 0) + 1
        dict_info["mean_rates"] += feedback
        dict_info["feedback"].setdefault(user, {}).update({item: feedback})
        dict_info["du"].setdefault(user, set()).add(item)
        dict_info["di"].setdefault(item, set()).add(user)

    dict_info["mean_rates"] /= dict_info["count"]
    return dict_info
