import argparse
import csv
import gc
import numpy as np
import os
import pickle as pkl
import sys
import time

from typing import Dict, Optional, Tuple

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)  # noqa: E402
from voting_rules_erosion.rule_loss_simulation_v2 import (
    initialize_all_voting_rules,
    PARAM_PAIR_FOR_DEBUG,
)


def transform_points_rule_to_duel_results(scores_data: np.ndarray) -> np.ndarray:
    """
    In every simulation, for each unique pair of choices (i, j), create the score attributed to i and j.
    Args:
        scores_data: an array of dimensions: num_simulations X num choices with the sum of scores per choice
    Returns:
        ndarray with dimensions: num_simulations X 2 (the pair) X Num distinct pairs
    """
    num_simulations, num_choices = scores_data.shape
    num_pairs = int(num_choices * (num_choices - 1) / 2)
    duel_scores = -np.ones((num_simulations, 2, num_pairs))
    ind = 0
    for i in range(num_choices):
        for j in range(i + 1, num_choices):
            duel_scores[:, 0, ind] = scores_data[:, i]
            duel_scores[:, 1, ind] = scores_data[:, j]
            ind += 1
    return duel_scores


def load_and_process_data_for_rule(
    rule_name: str,
    input_dir: str,
    distribution: str,
    num_simulations: int,
    n_c: int = None,
    n_v: int = None,
    analytical_results: bool = False,
    calculate_only_for_winners: bool = False,
) -> Tuple[dict, dict]:
    """
    Create the data path according to the input parameters, load the data file and transform
    data from point rules into the standard duel matrix form.
    If calculate_only_for_winners also load the winners data.
    Return:
        Dict of data in matrix duels format (dimensions: num_simulations X 2 X num_pairs) with
        two scores per unique choice pair.
        Optional: Dict of winners data with the index of the winner per simulation or Nan
        if no winner exists according to this rule.
    """
    winner_data_path: Optional[str] = None
    if n_c:
        # input data was saved per (n_v, n_c) pair
        if analytical_results:
            data_path = os.path.join(
                input_dir,
                f"{rule_name}_{n_v}_{n_c}_analytical_res.pkl",
            )
            if calculate_only_for_winners:
                winner_data_path = os.path.join(
                    input_dir,
                    f"{rule_name}_winner_{n_v}_{n_c}_analytical_res.pkl",
                )
        else:
            data_path = os.path.join(
                input_dir,
                f"{rule_name}_{n_v}_{n_c}_sims_{num_simulations}_{distribution}.pkl",
            )
            if calculate_only_for_winners:
                winner_data_path = os.path.join(
                    input_dir,
                    f"{rule_name}_winner_{n_v}_{n_c}_sims_{num_simulations}_{distribution}.pkl",
                )
    else:
        # input data was saved jointly for all pairs.
        data_path = os.path.join(
            input_dir, f"{rule_name}_sims_{num_simulations}_{distribution}.pkl"
        )
    if not os.path.isfile(data_path):
        raise ValueError(f"Data file {data_path} doesn't exist. Aborting")
    data = pkl.load(open(data_path, "rb"))
    all_rules = initialize_all_voting_rules()
    if all_rules[rule_name].is_point_rule:
        data = transform_points_rule_to_duel_results(data)
    winner_data = None
    if calculate_only_for_winners and winner_data_path is not None:
        if not os.path.isfile(winner_data_path):
            raise ValueError(f"No such file as {winner_data_path}")
        winner_data = pkl.load(open(winner_data_path, "rb"))
    return data, winner_data


def get_one_rule_erosion_data(
    data,
    erosion_normalizer,
    debug_data,
) -> Tuple[np.array, np.array, np.array]:
    """
    Create the erosion results for one rule.
    For each pair, erosion happens when the preferred option according to the voting
    rule isn't the one chosen by the actual rule used. The erosion is defined as:
    (Higher option score - lower option score) / normalizer.
    The normalizer is chosen according to the erosion_normalizer:
    "not_chosen": normalize by the higher option score. Meaning what proportion of the scores
    for this option was eroded.
    "all_options": normalize by the sum of the scores for this pair. Meaning, what proportion
    of the total scores for these two options was eroded.

    Args:
        data: duel matrix with the results per pair. Dimensions: num_simulations X 2 X num_pairs
        erosion_normalizer: str, "not_chosen"/"all_options"
        debug_data: choice preference per voter. Passed to enable debugging.
    Return:
        1. erosion: Erosion per pair. Dimensions: num_simulations X num_pairs
        2. chosen_ind: Chosen index per pair (0 or 1). Dimensions: num_simulations X num_pair
        3. inds_to_ignore: Boolean array of pair indices that should be ignored because of no clear winner.
           Dimensions: num_simulations X num_pair
    """
    data_max = data.max(axis=1)
    erosion = (data_max - data.min(axis=1)).astype("float64")
    if erosion_normalizer == "not_chosen":
        erosion /= data_max
    elif erosion_normalizer == "all_options":
        erosion /= data.sum(axis=1)
    chosen_ind = data.argmax(axis=1)
    inds_to_ignore = (
        np.fliplr(data).argmax(axis=1) != 1 - chosen_ind
    )  # pair scores are equal
    erosion[inds_to_ignore] = 0
    return erosion, chosen_ind, inds_to_ignore


def reshape_to_2d(vec: np.array) -> np.array:
    """
    Utility function to reshape a 1d ndarray to 2d with the second dimension = 1
    """
    assert len(vec.shape) == 1
    return vec.reshape([vec.size, 1])


def focus_data_on_winners(
    data1_winner: np.ndarray,
    data2_winner: np.ndarray,
    inds_to_ignore1: np.ndarray,
    inds_to_ignore2: np.ndarray,
    rule1_erosion: np.ndarray,
    rule2_erosion: np.ndarray,
    rule1_chosen: np.ndarray,
    rule2_chosen: np.ndarray,
    num_choices: int,
):
    """
    Focus the data on winners of voting rules.

    For each rule's erosion data remove the erosion data for all the pairs
    except for the pair: (index of rule1 winner, index of rule2 winner)
    Add to the indexes to ignore all the simulations where the either there
    was no winner according to one of the rules (data1_winner or data2_winner == np.nan),
    or that both rules chose the same winner (no erosion).

    Args:
        Dimensions: num_simulations
        data1_winner (np.ndarray): The winners from the first voting rule.
        data2_winner (np.ndarray): The winners from the second voting rule.

        Dimensions: num_simulations X num_pairs
        inds_to_ignore1 (np.ndarray): Indices to ignore for the first voting rule.
        inds_to_ignore2 (np.ndarray): Indices to ignore for the second voting rule.
        rule1_erosion (np.ndarray): Erosion data for the first voting rule.
        rule2_erosion (np.ndarray): Erosion data for the second voting rule.
        rule1_chosen (np.ndarray): Chosen indices for the first voting rule.
        rule2_chosen (np.ndarray): Chosen indices for the second voting rule.
        num_choices (int): The number of choices.

    Returns:
        Tuple of numpy arrays: A tuple containing the following processed data arrays:
        Dimensions: num_simulations X 1
        - inds_to_ignore1
        - inds_to_ignore2
        - rule1_erosion
        - rule2_erosion
        - rule1_chosen
        - rule2_chosen
    """
    num_samples = rule1_erosion.shape[0]
    assert (
        data1_winner.shape[0]
        == data2_winner.shape[0]
        == num_samples
        == rule1_chosen.shape[0]
    )
    # 2d version of the vector, needed for broadcasting later
    data1_winner_2d = data1_winner.reshape([data1_winner.size, 1])
    # 2d version of the vector, needed for broadcasting later
    data2_winner_2d = data2_winner.reshape([data2_winner.size, 1])
    # Ignore all cases where there was no winner according to one of the rules
    inds_to_ignore1 = np.logical_or(inds_to_ignore1, np.isnan(data1_winner_2d))
    inds_to_ignore2 = np.logical_or(inds_to_ignore2, np.isnan(data2_winner_2d))
    # Ignore all cases where both rules chose the same winner (no erosion)
    inds_to_ignore1 = np.logical_or(inds_to_ignore1, data1_winner_2d == data2_winner_2d)
    inds_to_ignore2 = np.logical_or(inds_to_ignore2, data1_winner_2d == data2_winner_2d)
    # We need the winner vectors to be set to some number for the next step to work
    # for all simulations. Arbitrarily, chose index 0 as the winner in cases there
    # was no winner for that rule -> is anyhow set to be ignored.
    data1_winner_2d[np.isnan(data1_winner_2d)] = 0
    data2_winner_2d[np.isnan(data2_winner_2d)] = 0

    # Create the matrix mapping the indices of each pair to their position
    # in the row with per pair scores. Positions that aren't expect to be
    # used are set to -1.
    num_pairs = int(num_choices * (num_choices - 1) / 2)
    index_matrix = -np.ones((num_pairs, num_pairs), dtype="int32")
    ind = 0
    for i in range(num_choices):
        index_matrix[i, i] = 0
        for j in range(i + 1, num_choices):
            index_matrix[i, j] = ind
            index_matrix[j, i] = ind
            ind += 1
    data1_winner_2d = data1_winner_2d.astype("int32")
    data2_winner_2d = data2_winner_2d.astype("int32")
    winnining_pair_index1 = index_matrix[
        data1_winner_2d, data2_winner_2d
    ]  # index of the winning pair in the per pair results list
    sample_index = reshape_to_2d(
        np.arange(len(data1_winner_2d))
    )  # running index of the simulation number
    # For each row in all the result arrays - choose only the winning pair
    inds_to_ignore1 = inds_to_ignore1[sample_index, winnining_pair_index1]
    inds_to_ignore2 = inds_to_ignore2[sample_index, winnining_pair_index1]
    rule1_erosion = rule1_erosion[sample_index, winnining_pair_index1]
    rule2_erosion = rule2_erosion[sample_index, winnining_pair_index1]
    rule1_chosen = rule1_chosen[sample_index, winnining_pair_index1]
    rule2_chosen = rule2_chosen[sample_index, winnining_pair_index1]

    return (
        inds_to_ignore1,
        inds_to_ignore2,
        rule1_erosion,
        rule2_erosion,
        rule1_chosen,
        rule2_chosen,
    )


def gather_two_rules_erosion_data(
    data1: np.array,
    data2: np.array,
    data1_winner: np.array,
    data2_winner: np.array,
    erosion_normalizer: str,
    num_choices: int,
    debug_data: Optional[dict] = None,
):
    """
    Load the data for each rule and return arrays with the erosion and the index
    chosen for all pairs.
    If the winners data is provided, the data per simulation will be marginalized only to
    the following pair (rule1_winner, rule2_winner).

    In the case of only comparing the winner data only the results for the simulation
    profiles where both rules had a winner and the winners were different are relevant.
    To facilitate this, all other cases will be set as if both rules chose the same option.
    The results will include only the erosion and chosen index of the
    winning pair.
    Args:
        Dimensions: num_simulation X 2 X num_distinct_pairs
        data1 (np.array): Data for the first voting rule.
        data2 (np.array): Data for the second voting rule.
        Dimensions: num_simulations
        data1_winner (np.array): Winners for the first voting rule.
        data2_winner (np.array): Winners for the second voting rule.
        erosion_normalizer (str): Erosion normalizer type.
        num_choices (int): The number of choices.
        debug_data (Optional[dict]): Debug data for testing correctness including the
        preference profiles for all voters (default is None).

    Returns:
        Tuple of numpy arrays: A tuple containing the following processed data arrays:
        - rule1_chosen
        - rule2_chosen
        - rule1_erosion
        - rule2_erosion
        Dimensions in the default case: num_simulations X num_distinct_pairs
        Dimensions in the case of calculating the results over the pair of winners:
        num_simulations X 1
    """
    rule1_erosion, rule1_chosen, inds_to_ignore1 = get_one_rule_erosion_data(
        data1, erosion_normalizer, debug_data
    )
    rule2_erosion, rule2_chosen, inds_to_ignore2 = get_one_rule_erosion_data(
        data2, erosion_normalizer, debug_data
    )
    if data1_winner is not None:
        (
            inds_to_ignore1,
            inds_to_ignore2,
            rule1_erosion,
            rule2_erosion,
            rule1_chosen,
            rule2_chosen,
        ) = focus_data_on_winners(
            data1_winner,
            data2_winner,
            inds_to_ignore1,
            inds_to_ignore2,
            rule1_erosion,
            rule2_erosion,
            rule1_chosen,
            rule2_chosen,
            num_choices=num_choices,
        )
    # Equalize the chosen indexes for all the pairs to be ignored
    rule1_chosen[inds_to_ignore1] = rule2_chosen[inds_to_ignore1]
    rule2_chosen[inds_to_ignore2] = rule1_chosen[inds_to_ignore2]
    assert np.all(
        rule1_chosen[np.isnan(rule1_erosion)] == rule2_chosen[np.isnan(rule1_erosion)]
    )
    assert np.all(
        rule1_chosen[np.isnan(rule2_erosion)] == rule2_chosen[np.isnan(rule2_erosion)]
    )
    assert np.all(rule1_erosion[rule1_chosen != rule2_chosen] > 0)
    assert np.all(rule2_erosion[rule1_chosen != rule2_chosen] > 0)
    assert rule1_erosion.min() >= 0
    assert rule2_erosion.min() >= 0
    return rule1_chosen, rule2_chosen, rule1_erosion, rule2_erosion


def create_two_rules_basic_erosion_stats(
    rule1_chosen: np.array,
    rule2_chosen: np.array,
    rule1_erosion: np.array,
    rule2_erosion: np.array,
    analytical_results_profile_weights: np.array,
    num_simulations: int,
    debug_data: Optional[dict] = None,
):
    """
    Calculates erosion statistics by comparing the chosen indices and erosion data between two voting rules.
    The function returns a dictionary with various erosion metrics, such as ratios, differences, and percentages.
    If analytical results are provided, it takes profile weights into account. If not, it computes simulation results.


    Args:
        rule1_chosen (np.array): Chosen index per pair for the first voting rule.
        rule2_chosen (np.array): Chosen index per pair for the second voting rule.
        rule1_erosion (np.array): Erosion data for the first voting rule.
        rule2_erosion (np.array): Erosion data for the second voting rule.
        analytical_results_profile_weights (np.array): Weights for each profile, used for analytical results
        (or None for simulation results).
        num_simulations (int): Number of simulations.
        debug_data (Optional[dict]): Debug data for testing correctness (default is None).

    Returns:
        dict: A dictionary containing various erosion statistics, such as ratios, differences, percentages, etc.

    The returned dictionary includes the following metrics:
    - 'pct_pairs_with_erosion': Percentage of pairs with erosion.
    - 'pct_profiles_with_erosion': Percentage of profiles with erosion.
    - 'min_ratios': Minimum erosion ratios.
    - 'max_ratios': Maximum erosion ratios.
    - 'mean_ratios': Mean erosion ratios.
    - 'min_differences': Minimum erosion differences.
    - 'max_differences': Maximum erosion differences.
    - 'mean_differences': Mean erosion differences.
    - 'pct_pairs_better': Percentage of pairs where rule 1 erosion is better.
    - 'mean_erosion1': Mean erosion for the first rule.
    - 'mean_erosion2': Mean erosion for the second rule.

    Note: This function assumes that both rule1_chosen and rule2_chosen have the same shape and that profile_weights
    align with the profiles used for analytical results.
    """
    res = dict()
    inds_with_erosion = rule1_chosen != rule2_chosen
    if analytical_results_profile_weights is not None:
        # These are the results of the analytical calculation.
        # For every type of statistic we need to take into account that row i
        # represents profile_weights[i] instances in expectation.
        profile_weights1 = analytical_results_profile_weights.reshape(
            [analytical_results_profile_weights.size, 1]
        )  # needed for broadcasting in multiplication later
        total_profile_weights = analytical_results_profile_weights.sum()
        num_pairs_per_simulation = rule1_chosen.shape[1]
        total_pairs_weights = total_profile_weights * num_pairs_per_simulation
        total_profile_weights_involved_in_erosion = np.sum(
            (inds_with_erosion) * profile_weights1
        )
        res["pct_pairs_with_erosion"] = (
            # each pair in row i - is counted as profile_weights[i] instances
            np.sum((inds_with_erosion) * profile_weights1)
            / total_pairs_weights
        )
        res["pct_profiles_with_erosion"] = (
            (inds_with_erosion).any(axis=1)
            * analytical_results_profile_weights  # multiply each row by its weight
        ).sum() / total_profile_weights
        if res["pct_profiles_with_erosion"] > 0:
            normalized_profile_weight = (
                profile_weights1 / total_profile_weights_involved_in_erosion
            )
            ratios = rule1_erosion / rule2_erosion
            relevant_ratios = ratios[inds_with_erosion]
            res["min_ratios"] = np.min(relevant_ratios)
            res["max_ratios"] = np.max(relevant_ratios)
            res["mean_ratios"] = (ratios * normalized_profile_weight)[
                inds_with_erosion
            ].sum()  # Weighted average of the ratios
            differences = rule1_erosion - rule2_erosion
            relevant_differences = differences[inds_with_erosion]
            res["min_differences"] = np.min(relevant_differences)
            res["max_differences"] = np.max(relevant_differences)
            res["mean_differences"] = (differences * normalized_profile_weight)[
                inds_with_erosion
            ].sum()  # weighted average of the differences
            rule1_erosion_is_better = (
                rule1_erosion - rule2_erosion < 0
            ) * normalized_profile_weight
            res["pct_pairs_better"] = np.sum(
                rule1_erosion_is_better[inds_with_erosion]
            )  # weighted average of the number of cases rule 1 erosion was better
            rule1_erosion *= normalized_profile_weight
            rule2_erosion *= normalized_profile_weight
            erosion1 = rule1_erosion[inds_with_erosion]
            erosion2 = rule2_erosion[inds_with_erosion]
            assert np.all(erosion1 > 0)
            assert np.all(erosion2 > 0)
            res["mean_erosion1"] = np.sum(
                erosion1
            )  # The elements were already multiplied by the
            # normalization factor - so we only need to sum to get the weight average over erosion
            # cases
            res["mean_erosion2"] = np.sum(erosion2)
        else:
            res["mean_erosion1"] = 0
            res["mean_erosion2"] = 0
            res["pct_pairs_better"] = 0
    else:
        # Simulation results
        erosion1 = rule1_erosion[inds_with_erosion]
        erosion2 = rule2_erosion[inds_with_erosion]
        assert np.all(erosion1 > 0)
        assert np.all(erosion2 > 0)
        res["pct_pairs_with_erosion"] = np.sum(inds_with_erosion) / rule1_chosen.size
        res["pct_profiles_with_erosion"] = (inds_with_erosion).any(
            axis=1
        ).sum() / num_simulations
        if res["pct_pairs_with_erosion"] > 0:
            ratios = erosion1 / erosion2
            differences = erosion1 - erosion2
            res["mean_erosion1"] = np.mean(erosion1)
            res["mean_erosion2"] = np.mean(erosion2)
            res["mean_ratios"] = np.mean(ratios)
            res["min_ratios"] = np.min(ratios)
            res["max_ratios"] = np.max(ratios)
            res["mean_differences"] = np.mean(differences)
            res["min_differences"] = np.min(differences)
            res["max_differences"] = np.max(differences)
            res["pct_pairs_better"] = np.sum(erosion1 - erosion2 < 0) / erosion1.size
        else:
            res["mean_erosion1"] = 0
            res["mean_erosion2"] = 0
            res["pct_pairs_better"] = 0

    return res


def create_report_entries_for_specific_n_v_and_n_c(
    stats: Dict[dict],
    param_pair: Tuple[int, int],
    erosion_stats: dict,
    rule1_mr_erosion: dict,
    rule2_mr_erosion: dict,
) -> Dict[dict]:
    def my_round(x, round_factor):
        """
        The rounding function used to create report entries.
        Enables different types of precision according to the input
        number.
        """
        if x > 1:
            return round(x, 1)
        elif x < 0.0002:
            return round(x, 5)
        else:
            return float(f"{x:.3g}")

    round_factor = 1
    stats["pct_pairs_with_erosion"][param_pair] = my_round(
        erosion_stats["pct_pairs_with_erosion"], round_factor
    )
    stats["pct_profiles_with_erosion"][param_pair] = my_round(
        erosion_stats["pct_profiles_with_erosion"], round_factor
    )
    if erosion_stats["pct_pairs_with_erosion"] > 0:
        stats["ratio_of_means"][param_pair] = my_round(
            erosion_stats["mean_erosion1"] / erosion_stats["mean_erosion2"],
            round_factor,
        )
        stats["mean_ratios"][param_pair] = (
            f"{my_round(erosion_stats['mean_ratios'], round_factor)}"
            f" ({my_round(erosion_stats['min_ratios'], round_factor)}:"
            f"{my_round(erosion_stats['max_ratios'], round_factor)})"
        )

        stats["mean_differences"][param_pair] = (
            f"{my_round(erosion_stats['mean_differences'], round_factor)}"
            f" ({my_round(erosion_stats['min_differences'], round_factor)}:"
            f"{my_round(erosion_stats['max_differences'], round_factor)})"
        )
        stats["pct_pairs_better"][param_pair] = my_round(
            erosion_stats["pct_pairs_better"], round_factor
        )
    else:
        for stat in stats:
            stats[stat][param_pair] = "Nan"

    if rule1_mr_erosion:
        assert rule2_mr_erosion
        if (
            rule1_mr_erosion["mean_erosion2"] == 0
            and rule2_mr_erosion["mean_erosion2"] == 0
        ):
            stats["ratio_mean_MR_erosion"][param_pair] = "Nan"
            stats["ratio_pct_profiles_MR_erosion"][param_pair] = "Nan"
        elif rule1_mr_erosion["mean_erosion2"] == 0:
            stats["ratio_mean_MR_erosion"][param_pair] = 0
            stats["ratio_pct_profiles_MR_erosion"][param_pair] = 0
        elif rule2_mr_erosion["mean_erosion2"] == 0:
            stats["ratio_mean_MR_erosion"][param_pair] = "Inf"
            stats["ratio_pct_profiles_MR_erosion"][param_pair] = "Inf"
        else:
            stats["ratio_mean_MR_erosion"][
                param_pair
            ] = f"{rule1_mr_erosion['mean_erosion2'] / rule2_mr_erosion['mean_erosion2']}"
            stats["ratio_pct_profiles_MR_erosion"][
                param_pair
            ] = f"{rule1_mr_erosion['pct_profiles_with_erosion'] / rule2_mr_erosion['pct_profiles_with_erosion']}"
        stats["difference_mean_MR_erosion"][
            param_pair
        ] = f"{rule1_mr_erosion['mean_erosion2'] - rule2_mr_erosion['mean_erosion2']}"
        stats["difference_pct_profiles_MR_erosion"][
            param_pair
        ] = f"{rule1_mr_erosion['pct_profiles_with_erosion'] - rule2_mr_erosion['pct_profiles_with_erosion']}"


def create_erosion_stats_reports_for_two_rules(
    *,
    rule1_name: str,
    rule2_name: str,
    input_dir: str,
    distribution: str,
    num_simulations: int,
    erosion_normalizer: str,
    analytical_results: bool,
    output_dir: str,
    calculate_only_for_winners: bool = False,
    debug: bool = False,
):
    """
    Create csv files with different stats of the erosion results of two voting rules
    """
    all_rules = initialize_all_voting_rules()
    if rule1_name not in all_rules:
        raise ValueError(
            f"Rule name {rule1_name} isn't supported.\nAvailable options: {all_rules.keys()}"
        )
    if rule2_name not in all_rules:
        raise ValueError(
            f"Rule name {rule2_name} isn't supported.\nAvailable options: {all_rules.keys()}"
        )
    print(f"Creating stats for {rule1_name} and {rule2_name}")

    # Prepare the different stats dictionaries.
    # The dictionary keys are all the (n_v, n_c) combination pairs.
    stats_report = {
        "ratio_of_means": dict(),
        "mean_ratios": dict(),
        "mean_differences": dict(),
        "pct_pairs_better": dict(),
        "pct_pairs_with_erosion": dict(),
        "pct_profiles_with_erosion": dict(),
    }
    if all_rules[rule1_name].is_point_rule and all_rules[rule2_name].is_point_rule:
        stats_report["ratio_mean_MR_erosion"] = dict()
        stats_report["difference_mean_MR_erosion"] = dict()
        stats_report["ratio_pct_profiles_MR_erosion"] = dict()
        stats_report["difference_pct_profiles_MR_erosion"] = dict()

    # Find the relevant simulation files in the input directory
    files_in_dir = os.listdir(input_dir)
    files_for_rule1 = [
        f
        for f in files_in_dir
        if f.startswith(rule1_name)
        and not f.startswith(f"{rule1_name}_w")
        and f.split("_")[1].isdigit()
    ]
    files_for_rule2 = [
        f
        for f in files_in_dir
        if f.startswith(rule2_name)
        and not f.startswith(f"{rule2_name}_w")
        and f.split("_")[1].isdigit()
    ]
    parameter_pairs_for_rule1 = [
        (int(f.split("_")[1]), int(f.split("_")[2])) for f in files_for_rule1
    ]
    parameter_pairs_for_rule2 = [
        (int(f.split("_")[1]), int(f.split("_")[2])) for f in files_for_rule2
    ]
    assert sorted(parameter_pairs_for_rule1) == sorted(parameter_pairs_for_rule2)

    debug_data = None
    if debug:
        # Only a specific set pair will be debugged
        parameter_pairs_for_rule1 = parameter_pairs_for_rule2 = [PARAM_PAIR_FOR_DEBUG]
        debug_data_path = os.path.join(
            input_dir,
            f"debug_{PARAM_PAIR_FOR_DEBUG[0]}_{PARAM_PAIR_FOR_DEBUG[1]}_sims_{num_simulations}_{distribution}.pkl",
        )
        debug_data = pkl.load(open(debug_data_path, "rb"))

    # Gather erosion stats for each (n_v, n_c) parameter pair
    for param_pair in parameter_pairs_for_rule1:
        n_v, n_c = param_pair
        if analytical_results and param_pair == (9, 4):
            continue
        # load all relevant data
        data1, data1_winner = load_and_process_data_for_rule(
            rule_name=rule1_name,
            input_dir=input_dir,
            distribution=distribution,
            num_simulations=num_simulations,
            n_v=n_v,
            n_c=param_pair[1],
            analytical_results=analytical_results,
            calculate_only_for_winners=calculate_only_for_winners,
        )
        data2, data2_winner = load_and_process_data_for_rule(
            rule_name=rule2_name,
            input_dir=input_dir,
            distribution=distribution,
            num_simulations=num_simulations,
            n_v=n_v,
            n_c=param_pair[1],
            analytical_results=analytical_results,
            calculate_only_for_winners=calculate_only_for_winners,
        )

        analytical_results_profile_weights = None
        if analytical_results:
            # The analytical results contain one row for each voters profile combination.
            # The profile weights vector contains the number of occurrences of every such profile.
            profile_weights_path = os.path.join(
                input_dir,
                f"profile_weight_{n_v}_{n_c}_analytical_res.pkl",
            )
            analytical_results_profile_weights = pkl.load(
                open(profile_weights_path, "rb")
            )

        num_simulations = data1.shape[0]
        (
            rule1_chosen,
            rule2_chosen,
            rule1_erosion,
            rule2_erosion,
        ) = gather_two_rules_erosion_data(
            data1=data1,
            data2=data2,
            data1_winner=data1_winner,
            data2_winner=data2_winner,
            erosion_normalizer=erosion_normalizer,
            num_choices=n_c,
            debug_data=debug_data,
        )
        erosion_stats = create_two_rules_basic_erosion_stats(
            rule1_chosen=rule1_chosen,
            rule2_chosen=rule2_chosen,
            rule1_erosion=rule1_erosion,
            rule2_erosion=rule2_erosion,
            analytical_results_profile_weights=analytical_results_profile_weights,
            num_simulations=num_simulations,
            debug_data=debug_data,
        )

        rule1_mr_erosion = rule2_mr_erosion = None
        # If both rules are point rules, load the MR rule data and calculate the MR erosion for each point rule
        if all_rules[rule1_name].is_point_rule and all_rules[rule2_name].is_point_rule:
            MR_data, MR_data_winner = load_and_process_data_for_rule(
                rule_name="MR",
                input_dir=input_dir,
                distribution=distribution,
                num_simulations=num_simulations,
                n_v=n_v,
                n_c=n_c,
                analytical_results=analytical_results,
                calculate_only_for_winners=calculate_only_for_winners,
            )
            rule1_mr_erosion = create_two_rules_basic_erosion_stats(
                data1,
                MR_data,
                data1_winner=data1_winner,
                data2_winner=MR_data_winner,
                analytical_results_profile_weights=analytical_results_profile_weights,
                num_simulations=num_simulations,
                erosion_normalizer=erosion_normalizer,
                num_choices=n_c,
            )
            rule2_mr_erosion = create_two_rules_basic_erosion_stats(
                data2,
                MR_data,
                data1_winner=data2_winner,
                data2_winner=MR_data_winner,
                analytical_results_profile_weights=analytical_results_profile_weights,
                num_simulations=num_simulations,
                erosion_normalizer=erosion_normalizer,
                num_choices=n_c,
            )

        stats_report = create_report_entries_for_specific_n_v_and_n_c(
            stats=stats_report,
            param_pair=param_pair,
            erosion_stats=erosion_stats,
            rule1_mr_erosion=rule1_mr_erosion,
            rule2_mr_erosion=rule2_mr_erosion,
        )

    # Save each entry in the statistics report dictionary as a separate csv file
    for stat in stats_report:
        cur_res = stats_report[stat]
        if "MR_erosion" in stat:
            res_type = f"{stat}_{rule1_name}_{rule2_name}"
        else:
            # Swapping rule names because one rules erosion is actually the other's performance
            res_type = f"{stat}_{rule2_name}_{rule1_name}"
        res_type_text = res_type
        if calculate_only_for_winners:
            res_type_text += "_winners"
        if analytical_results_profile_weights is None:
            res_type_text += f"_{args.distribution}_dist_{args.num_simulations}_sims"
        csv_path = os.path.join(output_dir, res_type_text + ".csv")
        num_voters = sorted(list(set(x[0] for x in list(cur_res.keys()))))
        num_choices = sorted(list(set(x[1] for x in list(cur_res.keys()))))

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["num choices/num voters"] + num_voters)
            for cur_num_choices in num_choices:
                row = [cur_num_choices]
                row += [
                    cur_res[cur_num_voters, cur_num_choices]
                    if (cur_num_voters, cur_num_choices) in cur_res
                    else "Nan"
                    for cur_num_voters in num_voters
                ]
                writer.writerow(row)


if __name__ == "__main__":
    """
    Create erosion report from simulation data.
    The number of simulations and distribution type is needed in order to read correctly the input files.
    Can create all reports for the list of possible rule combinations or just for two specific rules.
    Can also create reports summarizing the results of the analytical calculation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument(
        "-d",
        "--distribution",
        type=str,
        default="cubic",
        help="cubic / box (euclidean box) / IANC",
    )
    parser.add_argument(
        "-n",
        "--num_simulations",
        type=int,
        default=1000,
        help="num simulations to perform",
    )
    parser.add_argument(
        "-a",
        "--rule1_name",
        type=str,
        default="MR",
        help="Choose from: MR, BR, PR",
    )
    parser.add_argument(
        "-b",
        "--rule2_name",
        type=str,
        default="BR",
        help="Choose from: MR, BR, PR. Needs to be different than rule1",
    )
    parser.add_argument(
        "-l",
        "--erosion_normalizer",
        type=str,
        default="not_chosen",
        help='How to normalize the erosion per option pair. One of: "not chosen" (the preference score for the option not'
             'chosen) / "all_options" (the sum of the preference scores for both options)',
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="/home/hi-auto/Pictures/social_choice"
    )
    parser.add_argument(
        "-all",
        "--create_reports_for_all_voting_rules",
        action="store_true",
        default=False,
        help="store_true",
    )
    parser.add_argument(
        "-dbg",
        "--debug",
        action="store_true",
        default=False,
        help="load debug data to be able to check correctness",
    )
    parser.add_argument(
        "-ana",
        "--analytical_results",
        help="Results are from anaylitical calculation and not from simulation. Requires different calculations",
        action="store_true",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input dir {args.input_dir} doesn't exist. Aborting")
    if not os.path.isdir(args.output_dir):
        raise ValueError(f"Output dir {args.output_dir} doesn't exist. Aborting")

    if args.create_reports_for_all_voting_rules:
        report_pairs = [
            ("MR", "BR"),
            ("MR", "PR"),
        ]
        for report_pair in report_pairs:
            t = time.time()
            cur_output_dir = os.path.join(
                args.output_dir,
                f'{report_pair[0]}_{report_pair[1]}',
            )
            os.makedirs(cur_output_dir, exist_ok=True)
            create_erosion_stats_reports_for_two_rules(
                rule1_name=report_pair[0],
                rule2_name=report_pair[1],
                input_dir=args.input_dir,
                output_dir=cur_output_dir,
                distribution=args.distribution,
                num_simulations=args.num_simulations,
                erosion_normalizer=args.erosion_normalizer,
                analytical_results=args.analytical_results,
                calculate_only_for_winners=False, # calculate_only_for_winners,
                debug=args.debug,
            )
            gc.collect()  # make sure the memory is cleared
            print(
                f"creating stats for {report_pair[0]} and {report_pair[1]} took {time.time() - t} seconds ({(time.time() - t)/60} minutes)"
            )
    else:
        t = time.time()
        rule1_name = args.rule1_name
        rule2_name = args.rule2_name
        output_dir = os.path.join(args.output_dir, f"{rule1_name}_{rule2_name}")
        os.makedirs(output_dir, exist_ok=True)
        create_erosion_stats_reports_for_two_rules(
            rule1_name=rule1_name,
            rule2_name=rule2_name,
            input_dir=args.input_dir,
            output_dir=output_dir,
            distribution=args.distribution,
            num_simulations=args.num_simulations,
            erosion_normalizer=args.erosion_normalizer,
            analytical_results=args.analytical_results,
            calculate_only_for_winners=False, # args.calculate_only_for_winners,
            debug=args.debug,
        )
        print(
            f"creating stats for {rule1_name} and {rule2_name} took {time.time() - t} seconds ({(time.time() - t)/60} minutes)"
        )
