from evaluation.item_recommendation import ItemRecommendationEvaluation
from evaluation.statistical_analysis import statistical_analysis

dir_results = "C:/Users/Arthur/Google Drive/Freelancer/Projetos/folds/"
baseline1 = "BPRMF_Mapping_ensemble_all_ensembles.dat"
baseline2 = "BPRMF_Mapping_ensemble_all_metadata.dat"
test_name = "test.dat"

result_baseline1 = ItemRecommendationEvaluation(only_map=True)
array_results1 = result_baseline1.folds_evaluation(dir_results, 10, baseline1, test_name, "AllButOne",
                                                   no_desviation=True)
result_baseline2 = ItemRecommendationEvaluation(only_map=True)
array_results2 = result_baseline2.folds_evaluation(dir_results, 10, baseline2, test_name, "AllButOne",
                                                   no_desviation=True)

for k, n in enumerate([1, 3, 5, 10]):
    print('MAP@' + str(n))
    statistical_analysis(array_results1[k], array_results2[k])
    print('\n')
