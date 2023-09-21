import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import anomaly_tpp as tpp

from tqdm.auto import tqdm, trange
from statsmodels.distributions.empirical_distribution import ECDF
sns.set_style('whitegrid')


num_sequences = 1000
t_max = 100
scenarios = [
    tpp.scenarios.spp.IncreasingRate(t_max),
    tpp.scenarios.spp.DecreasingRate(t_max),
    tpp.scenarios.spp.InhomogeneousPoisson(t_max),
    tpp.scenarios.spp.RenewalUp(t_max),
    tpp.scenarios.spp.RenewalDown(t_max),
    tpp.scenarios.spp.Hawkes(t_max),
    tpp.scenarios.spp.SelfCorrecting(t_max),
    tpp.scenarios.spp.Stopping(t_max),
]


test_statistics = [
    tpp.statistics_initial.ks_arrival,
    tpp.statistics_initial.ks_interevent,
    tpp.statistics_initial.chi_squared,
    tpp.statistics_initial.sum_of_squared_spacings,
    tpp.statistics_initial.Q_plus_statistic,
    tpp.statistics_initial.Q_minus_statistic,
]


# ### Estimate distribution of each test statistic under $H_0$

model = tpp.models.StandardPoissonProcess()
# in-distribution (ID) training sequences are used to estimate the CDF of the test statistic under H_0
# (this is then used to compute the p-values)
id_train = scenarios[0].sample_id(num_sequences)
id_train_batch = tpp.data.Batch.from_list(id_train)
id_train_poisson_times = tpp.utils.extract_poisson_arrival_times(model, id_train_batch)


# Empirical distribution of each test statistic on id_train.
# This approximates the CDF of the test statistic under H_0
# and is used to compute the p-values
ecdfs = {}

for stat in test_statistics:
    name = stat.__name__
    scores = stat(poisson_times_per_mark=id_train_poisson_times)
    ecdfs[name] = ECDF(scores)

def twosided_pval(stat_name: str, scores: np.ndarray):
    """Compute two-sided p-value for the given values of test statistic.
    
    Args:
        stat_name: Name of the test statistic, 
            {"ks_arrival", "ks_interevent", "chi_squared", "sum_of_squared_spacings", "Q_plus_statistic", "Q_minus_statistic"}
        scores: Value of the statistic for each sample in the test set,
            shape [num_test_samples]
    
    Returns:
        p_vals: Two-sided p-value for each sample in the test set,
            shape [num_test_samples]
    """
    ecdf = ecdfs[stat_name](scores)
    return 2 * np.minimum(ecdf, 1 - ecdf)


# ### Compute test statistic for ID test sequences

# ID test sequences will be compared to OOD test sequences to evaluate different test statistics
id_test = scenarios[0].sample_id(num_sequences)
id_test_batch = tpp.data.Batch.from_list(id_test)
id_test_poisson_times = tpp.utils.extract_poisson_arrival_times(model, id_test_batch)

# Compute the statistics for all ID test sequences
id_test_scores = {}
for stat in test_statistics:
    name = stat.__name__
    id_test_scores[name] = stat(poisson_times_per_mark=id_test_poisson_times)


# ### Compute test statistic for OOD test sequences & evaluate AUC ROC based on the p-values

results = []

detectability_values = np.arange(0, 0.95, step=0.05)
num_seeds = 10
for seed in trange(num_seeds):
    for scenario in tqdm(scenarios):
        for det in detectability_values:
            np.random.seed(seed)
            ood_test = scenario.sample_ood(num_sequences=num_sequences, detectability=det)
            ood_test_batch = tpp.data.Batch.from_list(ood_test)
            ood_poisson_times_per_mark = tpp.utils.extract_poisson_arrival_times(model, ood_test_batch)

            for stat in test_statistics:
                stat_name = stat.__name__
                id_scores = id_test_scores[stat_name]
                id_pvals = twosided_pval(stat_name, id_scores)

                ood_scores = stat(poisson_times_per_mark=ood_poisson_times_per_mark)
                ood_pvals = twosided_pval(stat_name, ood_scores)
                
                auc = tpp.utils.roc_auc_from_pvals(id_pvals, ood_pvals)

                res = {"statistic": stat_name, "seed": seed, "detectability": det, 
                       "auc": auc, "scenario": scenario.name}
                results.append(res)


df = pd.DataFrame(results)

for scen in df.scenario.unique():
    plt.figure(figsize=[4, 3], dpi=100)
    df_sub = df[df.scenario == scen]
    sns.pointplot(data=df_sub, x="detectability", y="auc", hue="statistic")
    ax = plt.gca()
    ax.set_xticks([1, 5, 10, 15, 18])
    ax.set_title(scen)
    plt.legend(fontsize=8)
    plt.savefig("./output/output_spp/{}.png".format(scen), bbox_inches="tight")
    plt.show()

