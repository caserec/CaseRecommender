import getopt
import os

import time

import sys

from evaluation.item_recommendation import ItemRecommendationEvaluation
from recommenders.item_recommendation.ensemble_average_based import EnsembleAverageBased
from recommenders.item_recommendation.ensemble_bpr_learning import EnsembleLearningBPR
from recommenders.item_recommendation.ensemble_tag_based import EnsembleTagBased
from utils.split_dataset import SplitDataset

top_n = 1000


# Recommender Algorithms [MyMediaLite 3.10]

def bprmf(dir_folds, interaction_file, interaction_rank):
    for fold in xrange(10):
        train_file = dir_folds + str(fold) + "\\" + interaction_file
        rank = dir_folds + str(fold) + "\\" + "rank_partial.dat"
        final_rank = dir_folds + str(fold) + "\\" + interaction_rank

        cmm1 = "mono C:\\MyMediaLite-3.10\\lib\\mymedialite\\item_recommendation.exe "
        cmm = cmm1 + "--training-file=" + str(train_file) + " --recommender=BPRMF --test-file=" + \
                     "C:\\MyMediaLite-3.10\\test.txt --prediction-file=" + str(rank) + \
                     " --random-seed=1  --predict-items-number=" + str(top_n)

        os.system(cmm)
        treat_output_bpr(rank, final_rank)
        os.remove(rank)


def svdplusplus(dir_folds, interaction_file, interaction_rank):
    for fold in xrange(10):
        train_file = dir_folds + str(fold) + "\\" + interaction_file
        rank = dir_folds + str(fold) + "\\" + "rank_partial.dat"
        test_file = dir_folds + str(fold) + "\\" + "test_svd.dat"
        final_rank = dir_folds + str(fold) + "\\" + interaction_rank

        cmm1 = "mono C:\\MyMediaLite-3.10\\lib\\mymedialite\\rating_prediction.exe "
        cmm = cmm1 + "--training-file=" + str(train_file) + " --recommender=MatrixFactorization --test-file=" + \
                     str(test_file) + " --prediction-file=" + str(rank)

        os.system(cmm)
        treat_output_svd(rank, final_rank)
        os.remove(rank)
        os.remove(test_file)

# Utils


def treat_output_bpr(bpr_output, final_raking):
    rank = list()
    with open(bpr_output) as infile:
        for line in infile:
            factors = line.split()
            f1 = factors[1].split(',')
            for i, item in enumerate(f1):
                a = item.split(':')
                a[0] = a[0].replace('[', '')
                a[1] = a[1].replace(']', '')

                rank.append((int(factors[0]), int(a[0]), float(a[1])))
    rank = sorted(rank, key=lambda x: (x[0], -x[2]))

    with open(final_raking, 'w') as infile3:
        for triple in rank:
            infile3.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')


def treat_output_svd(svd_output, final_rank):
    dict_items = dict()
    list_users = list()

    with open(svd_output) as infile:
        for line in infile:
            if line.strip():
                inline = line.split('\t')
                user, item, interaction = int(inline[0]), int(inline[1]), float(inline[2])
                dict_items.setdefault(user, {}).update({item: interaction})
                list_users.append(user)

    list_users = sorted(list(set(list_users)))
    list_all_items = list()
    for user in list_users:
        list_items = list()
        for item in dict_items[user]:
            list_items.append([item, float(dict_items[user][item])])
        list_items.sort(key=lambda x: -x[1])
        list_all_items.append(list_items)

    with open(final_rank, 'w') as infile_w:
        for i, user in enumerate(list_users):
            for item in list_all_items[i][:top_n]:
                infile_w.write(str(user) + '\t' + str(item[0]) + '\t' + str(item[1]) + '\n')


def create_test_svd(dir_folds, train_file):
    for fold in xrange(10):
        file_train = dir_folds + str(fold) + "\\" + train_file
        list_items = list()
        list_users = list()
        dict_user = dict()
        dict_not_seen_items = dict()
        with open(file_train) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split('\t')
                    user, item, interaction = int(inline[0]), int(inline[1]), float(inline[2])

                    dict_user.setdefault(user, {}).update({item: interaction})
                    list_items.append(item)
                    list_users.append(user)

        list_items = sorted(list(set(list_items)))
        list_users = sorted(list(set(list_users)))

        for user in list_users:
            dict_not_seen_items[user] = list(set(list_items) - set(dict_user[user]))

        test_file = dir_folds + str(fold) + "\\" + "test_svd.dat"
        with open(test_file, 'w') as infile:
            for user in list_users:
                for item in dict_not_seen_items:
                    infile.write(str(user)+'\t'+str(item)+'\t1\n')

        del list_items
        del list_users
        del dict_user
        del dict_not_seen_items


def print_results(results_evaluation):
    list_labels = list()
    for i in [1, 3, 5, 10]:
        list_labels.append("Prec@" + str(i))
        list_labels.append("Recall@" + str(i))
        list_labels.append("MAP@" + str(i))

    for n, res in enumerate(results_evaluation):
        print(list_labels[n])
        print("Mean: " + str(res[0]))
        print("p-value: " + str(res[1]))
    print("\n")


def main(argv):
    # dir_path = "C:\\Users\\Arthur\\OneDrive\\Experimentos_2015.12"
    dir_path = ""

    try:
        opts, args = getopt.getopt(argv, "d:", ["dir_fold="])
    except getopt.GetoptError:
        print("Error: Please enter a valid directory!")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dir_fold"):
            dir_path = arg

    if dir_path == "":
        print("Error:  Please enter a valid directory!")
        sys.exit(2)

    # ---------- LastFM Dataset ----------
    print("Processing LastFM Dataset")
    print("Dividing dataset...")
    starting_point = time.time()

    # Divide Dataset

    history_file = dir_path + "\\lastfm\\dataset\\user_artists.dat"
    tag_file = dir_path + "\\lastfm\\dataset\\user_tag_artist.dat"
    dir_fold = dir_path + "\\lastfm\\"
    SplitDataset([history_file, tag_file], dir_folds=dir_fold)

    # Run baselines

    print("Run Baselines...")
    dir_folds_lastfm = dir_path + "\\lastfm\\folds\\"

    # Visualization History (BPR MF)

    history_train = "train_user_artists.dat"
    history_rank = "rank_history.dat"
    bprmf(dir_folds_lastfm, history_train, history_rank)

    # Tags (BPR MF)
    tag_train = "train_user_tag_artist.dat"
    tag_rank = "rank_tags.dat"
    bprmf(dir_folds_lastfm, tag_train, tag_rank)

    # Ensembles (BPR Learning / Tag-based / Average-based)

    for fold in xrange(1):
        path_dir = dir_folds_lastfm + str(fold) + "\\"
        file_write_bpr = path_dir + "rank_ensemble_bpr.dat"
        file_write_tag = path_dir + "rank_ensemble_tags.dat"
        file_write_average = path_dir + "rank_ensemble_average.dat"

        history_t = path_dir + history_train
        tag_t = path_dir + tag_train

        history_r = path_dir + history_rank
        tag_r = path_dir + tag_rank

        list_train_files = [history_t, tag_t]
        list_rank_files = [history_r, tag_r]

        EnsembleTagBased(list_train_files, list_rank_files, file_write_tag, rank_number=10, space_type='\t')
        EnsembleAverageBased(list_train_files, list_rank_files, file_write_average, rank_number=10, space_type='\t')
        EnsembleLearningBPR(list_train_files, list_rank_files, file_write_bpr, rank_number=10, space_type='\t')

    # Evaluation Results
    print("Evaluating results...\n ")
    results = ItemRecommendationEvaluation()
    # History
    list_results = results.folds_evaluation(dir_folds_lastfm, 10, history_rank, "test.dat", "AllButOne")
    print("History Ranking \n")
    print_results(list_results)
    # Tags
    list_results = results.folds_evaluation(dir_folds_lastfm, 10, tag_rank, "test.dat", "AllButOne")
    print("Tags Ranking \n")
    print_results(list_results)
    # Ensemble Tags-based
    list_results = results.folds_evaluation(dir_folds_lastfm, 10, "rank_ensemble_tags.dat.dat", "test.dat", "AllButOne")
    print("Ensemble Tags-based Ranking \n")
    print_results(list_results)
    # Ensemble Average-based
    list_results = results.folds_evaluation(dir_folds_lastfm, 10, "rank_ensemble_average", "test.dat", "AllButOne")
    print("Ensemble Average-based Ranking \n")
    print_results(list_results)
    # Ensemble Bpr-Learning
    list_results = results.folds_evaluation(dir_folds_lastfm, 10, "rank_ensemble_bpr.dat", "test.dat", "AllButOne")
    print("Ensemble BPR-Learning Ranking \n")
    print_results(list_results)

    elapsed_time = time.time() - starting_point
    print("\nRuntime: " + str(elapsed_time / 60) + " minute(s)")
    print("Finished LastFm!\n")

    # # ---------- MovieLens Dataset ----------
    print("Processing MovieLens Dataset")
    print("Dividing dataset...")
    starting_point = time.time()

    # Divide Dataset

    history_file = dir_path + "\\movielens\\dataset\\user_movies.dat"
    tag_file = dir_path + "\\movielens\\dataset\\user_tag_movies.dat"
    dir_fold = dir_path + "\\movielens\\"
    SplitDataset([history_file, tag_file], dir_folds=dir_fold)

    # Run Baselines

    print("Run Baselines...")
    dir_folds_movielens = dir_path + "\\movielens\\folds\\"

    # Visualization History (BPR MF)

    history_train = "train.dat"
    history_rank = "rank_history.dat"
    bprmf(dir_folds_movielens, history_train, history_rank)

    # Tags (BPR MF)

    tag_train = "train_user_tag_movies.dat"
    tag_rank = "rank_tags.dat"
    bprmf(dir_folds_movielens, tag_train, tag_rank)

    # Ratings (SVD++)

    ratings_train = "train_user_movies.dat"
    ratings_rank = "rank_ratings.dat"
    create_test_svd(dir_folds_movielens, ratings_train)
    svdplusplus(dir_folds_movielens, ratings_train, ratings_rank)

    for fold in xrange(10):
        path_dir = dir_folds_movielens + str(fold) + "\\"
        file_write_bpr = path_dir + "rank_ensemble_bpr.dat"
        file_write_tag = path_dir + "rank_ensemble_tags.dat"
        file_write_average = path_dir + "rank_ensemble_average.dat"

        history_t = path_dir + history_train
        tag_t = path_dir + tag_train
        ratings_t = path_dir + ratings_train

        history_r = path_dir + history_rank
        tag_r = path_dir + tag_rank
        ratings_r = path_dir + ratings_rank

        list_train_files = [history_t, tag_t, ratings_t]
        list_rank_files = [history_r, tag_r, ratings_r]

        EnsembleTagBased(list_train_files, list_rank_files, file_write_tag, rank_number=10, space_type='\t')
        EnsembleAverageBased(list_train_files, list_rank_files, file_write_average, rank_number=10, space_type='\t')
        EnsembleLearningBPR(list_train_files, list_rank_files, file_write_bpr, rank_number=10, space_type='\t')

    # Evaluation Results
    print("Evaluating results...\n ")
    results = ItemRecommendationEvaluation()

    # History
    list_results = results.folds_evaluation(dir_folds_movielens, 10, history_rank, "test.dat", "AllButOne")
    print("History Ranking \n")
    print_results(list_results)

    # Tags
    list_results = results.folds_evaluation(dir_folds_movielens, 10, tag_rank, "test.dat", "AllButOne")
    print("Tags Ranking \n")
    print_results(list_results)

    # Ratings
    list_results = results.folds_evaluation(dir_folds_movielens, 10, ratings_rank, "test.dat", "AllButOne")
    print("Ratings Ranking \n")
    print_results(list_results)

    # Ensemble Tags-based
    list_results = results.folds_evaluation(dir_folds_movielens, 10,
                                            "rank_ensemble_tags.dat.dat", "test.dat", "AllButOne")
    print("Ensemble Tags-based Ranking \n")
    print_results(list_results)

    # Ensemble Average-based
    list_results = results.folds_evaluation(dir_folds_movielens, 10, "rank_ensemble_average", "test.dat", "AllButOne")
    print("Ensemble Average-based Ranking \n")
    print_results(list_results)

    # Ensemble Bpr-Learning
    list_results = results.folds_evaluation(dir_folds_movielens, 10, "rank_ensemble_bpr.dat", "test.dat", "AllButOne")
    print("Ensemble BPR-Learning Ranking \n")
    print_results(list_results)

    elapsed_time = time.time() - starting_point
    print("\nRuntime: " + str(elapsed_time / 60) + " minute(s)")
    print("Finished MovieLens!\n")

if __name__ == "__main__":
    main(sys.argv[1:])
