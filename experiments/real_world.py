import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import torch
import torch.nn as nn
import anomaly_tpp as tpp

from tqdm.auto import tqdm, trange
sns.set_style("whitegrid")


t_max = 100
batch_size = 64
num_seeds = 5


results = []

for seed in trange(num_seeds):
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    scenario = tpp.scenarios.real_world.ServerLogs()
    # scenario = tpp.scenarios.real_world.STEAD()

    # ratio = 1
    # scenario.id_train.Compression(proportion=ratio)
    id_train = scenario.id_train
    id_test = scenario.id_test
    
    dl_train = id_train.get_dataloader(batch_size=batch_size, shuffle=True)

    # Fit a neural TPP model on the training ID sequences
    ntpp = tpp.utils.fit_ntpp_model(dl_train, num_marks=id_train.num_marks)


    test_statistics = [
        tpp.statistics.Q_plus_statistic,
        tpp.statistics.Q_minus_statistic,
    ]


    # ### Using the KCDF to estimate the distribution of each test statistic associated with the event type under $H_0$


    # in-distribution (ID) training sequences are used to estimate the CDF of the test statistic associated with the event type under H_0
    # (this is then used to compute the p-values)
    id_train_batch = tpp.data.Batch.from_list(id_train)
    id_train_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_train_batch)

    id_train_dic = {}
    K = len(id_train_poisson_times[0])
    # print("num_marks:", K)
    for j in range(K):
        temp = []
        for i in id_train_poisson_times:
            temp.append(i[j])
        id_train_dic[j] = temp


    # Kernel cumulative distribution function (KCDF) of each test statistic associated with the event type on id_train.
    # This approximates the CDF of the test statistic associated with the event type under H_0
    # and is used to compute the p-values

    kcdfs = {}

    for stat in test_statistics:
        name = stat.__name__
        kcdf = {}
        for k in range(K):
            scores = stat(poisson_times_per_mark=id_train_dic[k], model=ntpp, batch=id_train_batch)
            kcdf[k] = lambda x, scores=scores: tpp.utils.kernel_distribution_estimation(x, scores)
        kcdfs[name] = kcdf

    def twosided_pval(stat_name: str, scores: np.ndarray, k: int):
        """Compute two-sided p_k for the given values of test statistic associated with the event type k.
    
    Args:
        stat_name: Name of the test statistic, 
            {"Q_plus_statistic", "Q_minus_statistic"}
        scores: Value of the statistic of time-rescaled subsequences of type k for each sample in the test set,
            shape [num_test_samples]
        k: Type of the time-rescaled subsequence, k = 0, ..., K-1
    
    Returns:
        p_k: Two-sided p-value of time-rescaled subsequences of type k for each sample in the test set,
            shape [num_test_samples]
    """
        kcdf_vectorized = np.vectorize(kcdfs[stat_name][k])
        kcdf = kcdf_vectorized(scores)
        return 2 * np.minimum(kcdf, 1 - kcdf)


    # ### Compute test statistic associated with the event type for ID test sequences

    # ID test sequences will be compared to OOD test sequences to evaluate different test statistics
    id_test_batch = tpp.data.Batch.from_list(id_test)
    id_test_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_test_batch)

        
    id_test_dic = {}

    for j in range(K):
        temp = []
        for i in id_test_poisson_times:
            temp.append(i[j]) 
        id_test_dic[j] = temp

    # Compute the statistics for all ID test sequences
    id_test_scores = {}
    for stat in test_statistics:
        name = stat.__name__
        score = {}
        for k in range(K):
            scores = stat(poisson_times_per_mark=id_test_dic[k], model=ntpp, batch=id_test_batch)  
            score[k] = scores
        id_test_scores[name] = score

    # ### Compute test statistic associated with the event type for OOD test sequences & evaluate AUC ROC based on the Simes method corrected p-values
    for name, ood_test in scenario.ood_test_datasets.items():
        ood_test_batch = tpp.data.Batch.from_list(ood_test)
        ood_test_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, ood_test_batch)


        ood_test_dic = {}
        for j in range(K):
            temp = []
            for i in ood_test_poisson_times:
                temp.append(i[j])
            ood_test_dic[j] = temp


    # Compute the statistics for all OOD test sequences   
        ood_test_scores = {}
        for stat in test_statistics:
            stat_name = stat.__name__
            score = {}
            for k in range(K):
                scores = stat(poisson_times_per_mark=ood_test_dic[k], model=ntpp,batch=ood_test_batch)  
                score[k] = scores
            ood_test_scores[stat_name] = score

    # Evaluate AUC ROC based on the Simes method corrected p-values
        for stat in test_statistics:
            stat_name = stat.__name__
            id_scores = id_test_scores[stat_name]
            ood_scores = ood_test_scores[stat_name]

            id_pvals = {}
            ood_pvals = {}
            for k in range(K):
                id_pvals[k] = np.array(twosided_pval(stat_name, id_scores[k], k)) 
                ood_pvals[k] = np.array(twosided_pval(stat_name, ood_scores[k], k))
                
            
            id_pvals = np.array([id_pvals[k] for k in range(K)])
            id_pvals_sorted = np.sort(id_pvals, axis=0)
            indices = np.arange(1, K + 1).reshape(-1,1)
            id_pvals_corrected = id_pvals_sorted / indices * K

            ood_pvals = np.array([ood_pvals[k] for k in range(K)])
            ood_pvals_sorted = np.sort(ood_pvals, axis=0)
            indices = np.arange(1, K + 1).reshape(-1,1)
            ood_pvals_corrected = ood_pvals_sorted / indices * K

            id_pvals_corrected = np.minimum.reduce(id_pvals_corrected)
            ood_pvals_corrected = np.minimum.reduce(ood_pvals_corrected)
        
            auc = tpp.utils.roc_auc_from_pvals(id_pvals_corrected, ood_pvals_corrected)

            res = {"statistic": stat_name, "auc": auc, "scenario": name}
            results.append(res)

            # rec = tpp.utils.recall_score_from_pvals(id_pvals_corrected, ood_pvals_corrected)
            # f1 = tpp.utils.f1_score_from_pvals(id_pvals_corrected, ood_pvals_corrected)
            # res = {"statistic": stat_name, "auc": auc, "rec": rec, "f1": f1, "scenario": name}
            # results.append(res)


df = pd.DataFrame(results)

grouped_df = df.groupby(['scenario', 'statistic'])['auc'].agg(['mean', 'std']).reset_index()
grouped_df['mean'] = grouped_df['mean'].apply(lambda x: round(x * 100, 1))
grouped_df['std'] = grouped_df['std'].apply(lambda x: round(x * 100, 1))
grouped_df.to_csv('./output/output_real_world/ServerLogs.csv', index=False)

# grouped_df = df.groupby(['scenario', 'statistic'])[['auc', 'rec', 'f1']].agg('mean').sort_values(by=['scenario', 'statistic']).reset_index()
# grouped_df[['auc', 'rec', 'f1']] = grouped_df[['auc', 'rec', 'f1']].apply(lambda x: round(x * 100, 1))
# grouped_df.to_csv('./output/output_real_world/ServerLogs.csv', index=False)