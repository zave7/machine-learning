from typing import Iterable
import pandas as pd
import numpy as np

def bayesian_table(table: pd.DataFrame, prior: float | Iterable[float], likelihood: Iterable[float]) -> pd.DataFrame:
    print(f"likelihood = {likelihood}")
    calc_prior = (prior, 1-prior) if type(prior) is float else prior
    posterior = table.get("posterior")
    posterior = posterior if posterior is not None else calc_prior
    updated_table = pd.DataFrame(index=table.index)
    updated_table["prior"] = posterior
    updated_table["likelihood"] = likelihood
    updated_table["joint"] = posterior * updated_table["likelihood"]
    norm_const = updated_table["joint"].sum()
    updated_table["posterior"] = updated_table["joint"] / norm_const
    return updated_table

def bayesian_table_custom(prior: float | Iterable[float], likelihood: Iterable[float]) -> Iterable[float]:
    joint = prior * np.array(likelihood)
    norm_const = joint.sum()
    return joint / norm_const

def get_likelihoods(data: pd.DataFrame, hypotheses: list[str] | list[float], feature_name: str) -> dict:
    likelihoods = {}
    for hypothesis in hypotheses:
        indices = { hypothesis: data.iloc[:, -1] == hypothesis for hypothesis in data.iloc[:, -1].unique() }
        feature_data_in_hypothesis = data[feature_name][indices[hypothesis]]
        len_feature_data_in_hypothesis = len(feature_data_in_hypothesis)
        feature_all_states = np.unique(data[feature_name])
        (states, counts) = np.unique(feature_data_in_hypothesis, return_counts=True)
        curr_hypothesis_feature_state_dict = { state: counts[idx_state] for (idx_state, state) in enumerate(states) }
        for (_, state) in enumerate(feature_all_states):
            likelihood_list = likelihoods.get(state, [])
            count = curr_hypothesis_feature_state_dict.get(state, 0)
            likelihood_list.append(count / len_feature_data_in_hypothesis)
            likelihoods[state] = likelihood_list
    return likelihoods

# 