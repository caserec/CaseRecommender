from framework.utils import ReadFile
from framework.utils import WriteFile


class BaseEnsemble(object):
    def __init__(self, list_ranks, test_file, write_file=""):
        self.list_ranks = list_ranks
        self.test_file = test_file
        self.write_file = write_file
        self.dict_ranks = ReadFile(self.list_ranks).ensemble()
        self.list_users = set()
        self.final_ranking = dict()

        # methods
        self.ensemble()
        self.write_results()

    def ensemble(self):
        for rank in self.dict_ranks:
            for user in self.dict_ranks[rank]["users"]:
                self.list_users.add(user)
                for i, item in enumerate(self.dict_ranks[rank]["rank"][user]):
                    try:
                        self.final_ranking[item[0]][item[1]] += item[2]
                    except KeyError:
                        self.final_ranking.setdefault(user, {}).update({item[1]: item[2]})
        self.list_users = sorted(self.list_users)
        for user in self.list_users:
            self.final_ranking[user] = sorted(self.final_ranking[user].items(), key=lambda x: -x[1])[:10]

    def write_results(self):
        WriteFile(self.write_file, self.final_ranking).write_ensemble(self.list_users)


r1 = "C:/Users/Arthur/OneDrive/Experimentos_2015.12/movielens/folds/0/rank_history.dat"
r2 = "C:/Users/Arthur/OneDrive/Experimentos_2015.12/movielens/folds/0/rank_tags.dat"
test = "C:/Users/Arthur/OneDrive/Experimentos_2015.12/movielens/folds/0/test.dat"
wf = "C:/Users/Arthur/OneDrive/Experimentos_2015.12/movielens/folds/0/ensemble.dat"

BaseEnsemble([r1, r2], test, write_file=wf)
