"""
 Trying to make a new ensemble

 ruif = ruir1 * ar1u1

 var types

 train_files -> list
 rankings_files - > list

"""
import random

import operator


class NewEnsemble(object):
    def __init__(self, train_files, rankings_files, space_type='\t'):
        self.train_files = train_files
        self.rankings_files = rankings_files
        self.space_type = space_type

        # vars
        self.seen_items = dict()
        self.non_seen_items = dict()
        self.list_users = set()
        self.list_items = set()
        self.alpha = dict()
        self.rankings = dict()

        # methods
        self.reading_train_files()
        self.read_rankings()

    def normalize_rankings(self):
        pass

    # reading files
    def reading_train_files(self):
        for train_file in self.train_files:
            with open(train_file) as infile:
                for line in infile:
                    if line.strip():
                        inline = line.split(self.space_type)
                        self.seen_items.setdefault(int(inline[0]), set()).add(int(inline[1]))
                        self.list_users.add(int(inline[0]))
                        self.list_items.add(int(inline[1]))

        self.list_users = sorted(self.list_users)

        for user in self.list_users:
            self.non_seen_items.setdefault(user, list(set(self.list_items)-set(self.seen_items[user])))

    # def read_rankings(self):
    #     for r, ranking in enumerate(self.rankings_files):
    #         position = 0
    #         seen_items_rankings = dict()
    #         with open(ranking) as infile:
    #             for line in infile:
    #                 if line.strip():
    #                     inline = line.split(self.space_type)
    #                     if position > 10:
    #                         position = 0
    #                     else:
    #                         position += 1
    #                     seen_items_rankings.setdefault(int(inline[0]), {}).update({int(inline[1]): position})
    #                     self.rankings[r].setdefault(int(inline[0]), {}).update({int(inline[1]): float(inline[2])})
    #                     self.users_ranking_items.setdefault(int(inline[0]), {}).update({int(inline[1]): r})
    #
    #         # random method
    #         for user in self.list_users:
    #             b = list()
    #             for i in xrange(10):
    #                 choice_items = random.sample(set(self.non_seen_items[user]), int(len(self.list_items) * 0.1))
    #                 try:
    #                     b.append(len(set(seen_items_rankings[user]).intersection(choice_items))/float(10))
    #                 except KeyError:
    #                     b.append(-1)
    #             if max(b) != 0.0:
    #                 self.alpha.setdefault(r, []).append(max(b))
    #             else:
    #                 self.alpha.setdefault(r, []).append(0.00000000001)
    #
    # def build_new_ranking(self):
    #     for u, user in enumerate(self.list_users):
    #         rui = dict()
    #         for item in self.users_ranking_items[user]:
    #             # print self.alpha[self.users_ranking_items[user][item]]
    #             # print self.rankings[self.users_ranking_items[user][item]][user][item]
    #             try:
    #                 print rui[user][item]
    #                 rui[user][item] += (self.alpha[self.users_ranking_items[user][item]][u] *
    #                                     self.rankings[self.users_ranking_items[user][item]][user][item])
    #                 print 'ouhhuuo'
    #
    #             except KeyError:
    #                 # print self.rankings[self.users_ranking_items[user][item]][user][item]
    #                 # print self.alpha[self.users_ranking_items[user][item]][u]
    #                 rui.setdefault(user, {}).update({item: self.alpha[self.users_ranking_items[user][item]][u] *
    #                                                 self.rankings[self.users_ranking_items[user][item]][user][item]})
    #
    #         rank = sorted(rui[user].items(), key=operator.itemgetter(1), reverse=True)[:10]
    #         with open(ranking_write, 'a') as inf_write:
    #             for item in rank:
    #                 inf_write.write(str(user) + self.space_type + str(item[0]) +
    # self.space_type + str(item[1]) + "\n")

    def read_rankings(self):
        n_samples = int(len(self.list_items) * 0.1)
        actual_user = -1
        all_items_in_ranking = dict()

        for r, ranking in enumerate(self.rankings_files):
            list_users_rankings = set()
            seen_items_rankings = dict()
            user_ranking = list()
            user_items = list()
            scores = dict()

            with open(ranking) as infile:
                for line in infile:
                    if line.strip():
                        inline = line.split(self.space_type)
                        seen_items_rankings.setdefault(int(inline[0]), []).append(int(inline[1]))
                        all_items_in_ranking.setdefault(int(inline[0]), set()).add(int(inline[1]))
                        list_users_rankings.add(int(inline[0]))

                        if len(user_ranking) < 10:
                            user_ranking.append(float(inline[2]))
                            user_items.append(int(inline[1]))
                            actual_user = int(inline[0])
                        else:
                            user_ranking = [float(i) / max(user_ranking) for i in user_ranking]
                            final_ranking = list()
                            [final_ranking.append([user_items[i], user_ranking[i]]) for i in xrange(10)]
                            scores.setdefault(actual_user, final_ranking)
                            user_ranking = [float(inline[2])]
                            user_items = [int(inline[1])]

            list_users_rankings = sorted(list_users_rankings)

            # # new random
            for user in list_users_rankings:
                print user
                self.rankings.setdefault(user, {})
                c = 0
                for i in xrange(100):
                    hidden_item = random.choice(list(self.seen_items[user]))
                    new_set = self.non_seen_items[75]
                    new_set.append(hidden_item)
                    choice_items = random.sample(set(new_set), n_samples)
                    y = 0 if hidden_item not in choice_items else 1
                    precision_partial = len(set(seen_items_rankings[user]).intersection(choice_items)) / float(10)
                    c += (y * precision_partial)

                alpha = c / float(100)
                try:
                    partial_ranking = [x[1] * alpha for x in scores[user]]
                    for k, item in enumerate(scores[user]):
                        if item[0] in self.rankings[user]:
                            self.rankings[user][item[0]] += partial_ranking[k]
                        else:
                            self.rankings[user].update({item[0]: partial_ranking[k]})
                except KeyError:
                    pass

        for user in self.list_users:
            final_results = sorted(self.rankings[user].items(), key=operator.itemgetter(1), reverse=True)[:10]
            print final_results

            # with open(self.ranking_write, 'a') as inf_write:
            #     for item in final_results:
            #     inf_write.write(str(user) + self.space_type + str(item[0]) + self.space_type + str(item[1]) + "\n")
