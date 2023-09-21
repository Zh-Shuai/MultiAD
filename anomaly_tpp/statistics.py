from typing import List

import numpy as np
import torch


__all__ = [
    "Q_plus_statistic",
    "Q_minus_statistic",
]

def Q_plus_statistic(poisson_times_per_mark: List[np.ndarray], **kwargs):

    def soss_single_sequence(t: np.ndarray):
        deltas = np.ediff1d(np.concatenate([[0], t]))
        N = len(deltas)
        scores_1 = np.linalg.norm(deltas) ** 2
        scores_2 = sum([deltas[i]*deltas[i+1] for i in range(N-1)])
        scores = (scores_1 + scores_2) / t[-1]
        return scores
    
    return np.array([soss_single_sequence(t) for t in poisson_times_per_mark])

def Q_minus_statistic(poisson_times_per_mark: List[np.ndarray], **kwargs):

    def soss_single_sequence(t: np.ndarray):
        deltas = np.ediff1d(np.concatenate([[0], t]))
        N = len(deltas)
        scores_1 = np.linalg.norm(deltas) ** 2
        scores_2 = sum([deltas[i]*deltas[i+1] for i in range(N-1)])
        scores = (scores_1 - scores_2) / t[-1]
        return scores
    
    return np.array([soss_single_sequence(t) for t in poisson_times_per_mark])