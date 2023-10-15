import argparse
import os
import pickle as pkl
import time

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from svvamp import (
    GeneratorProfileCubicUniform,
    GeneratorProfileEuclideanBox,
    GeneratorProfileIanc,
    Profile,
)
from svvamp.utils.misc import initialize_random_seeds
from tqdm.contrib.concurrent import process_map

PARAM_PAIR_FOR_DEBUG = (5, 3)


@dataclass
class VotingRule:
    """
    Represents a voting rule.

    Attributes:
        name (str): The name of the voting rule.
        is_point_rule (bool): Indicates whether the rule is a point rule (True) or not (False).
        profile_object_name (str): The name of the profile object associated with the rule in
        the svvamp package.
    """

    name: str
    is_point_rule: bool
    profile_object_name: str

    def saved_row_size(self, num_choices):
        """
        Calculate the size of the saved row for the rule, based on the number of choices.

        Args:
            num_choices (int): Number of choices in the election.

        Returns:
            int: Size of the saved row.
        """
        if self.is_point_rule:
            return num_choices
        else:
            return int(num_choices * (num_choices - 1) / 2)


def initialize_all_voting_rules() -> Dict[str, VotingRule]:
    """
    Initialize and return a dictionary of all supported voting rules.

    Returns:
        Dict[str, VotingRule]: A dictionary with rule names as keys and VotingRule objects as values.
    """
    all_rules = {
        "MR": VotingRule(
            name="MR", is_point_rule=False, profile_object_name="matrix_duels_rk"
        ),
        "PR": VotingRule(
            name="PR", is_point_rule=True, profile_object_name="plurality_scores_rk"
        ),
        "BR": VotingRule(
            name="BR", is_point_rule=True, profile_object_name="borda_score_c_rk"
        ),
    }
    return all_rules


def generate_one_rule_results_for_one_profile(profile: Profile, rule: VotingRule):
    """
    Generate results for a specific voting rule applied to a profile.
    For point rules we will save the total scores of each of the choices (a vector of length num choices).
    For the majority rule we will keep the duel matrix for every unique pair of choices (i,j). The output
    will have the following dimensions: 2 (the pair) X Num distinct pairs for each simulation.
    For all rules we will save the index of the winning choice. If no clear one winner exists the value of the
    winner will be np.nan
    Args:
        profile (svvamp.Profile): The election profile.
        rule (VotingRule): The voting rule to apply.

    Returns:
        tuple: A tuple containing the simulation results and the winner for the rule.
    """
    num_choices = profile.n_c

    voting_rule_res = profile.__getattribute__(rule.profile_object_name)
    winner = np.nan
    if rule.is_point_rule:
        simulation_result = voting_rule_res
        winners = np.argwhere(voting_rule_res == max(voting_rule_res))
        if len(winners) == 1:
            winner = winners[0][0]
    else:
        simulation_result = np.zeros([2, rule.saved_row_size(num_choices=num_choices)])
        ind = 0
        for i in range(num_choices):
            for j in range(i + 1, num_choices):
                simulation_result[:, ind] = [
                    voting_rule_res[i, j],
                    voting_rule_res[j, i],
                ]
                assert voting_rule_res[i, j] + voting_rule_res[j, i] == profile.n_v
                ind += 1
        winner = profile.condorcet_winner_rk
    return simulation_result, winner


def simulate_one_profile(
    random_sample: Union[
        GeneratorProfileCubicUniform, GeneratorProfileEuclideanBox, GeneratorProfileIanc
    ],
    all_rules: List[VotingRule],
):
    """
    Generate a profile and return the simulation results for all voting rules.

    Args:
        random_sample (function): A function to generate a random profile.
        all_rules (List[VotingRule]): List of all voting rules.

    Returns:
        dict: A dictionary containing the results for each voting rule and their respective winners.
    """
    profile = random_sample()
    result = dict()
    for rule_name in all_rules:
        (
            result[rule_name],
            result[f"{rule_name}_winner"],
        ) = generate_one_rule_results_for_one_profile(profile, all_rules[rule_name])
    # result["MR_weak_winners"] = profile.weak_condorcet_winners
    return result


def simulate_voting_for_specific_nc_nv(
    all_rules: Dict[str, VotingRule],
    n_v: int,
    n_c: int,
    num_simulations: int,
    distribution: str,
    num_workers: int,
    output_dir: str,
    debug: bool = False,
):
    """
    Simulate voting for a specific combination of the number of voters and choices.

    This function generates election profiles and simulates voting for different voting rules. The simulation results
    are saved to files for analysis. For each combination of the number of voters and choices, the function gathers
    results for all voting rules present in 'all_rules'.
    For point rules the total scores of each of the choices (a vector of length num choices) will be saved.
    For the majority rule the duel matrix for every unique pair of choices (i,j) will be saved. The output
    will have the following dimensions: (the pair) X Num distinct pairs (per simulation)
    For all rules we will save the index of the winning choice. If no clear one winner exists the value of the
    winner will be np.nan
    Once all the simulation results are saved they can be analyzed with the create_stats.py script


    Args:
        all_rules (Dict[str, VotingRule]): A dictionary of all voting rules to be simulated.
        n_v (int): Number of voters in the election.
        n_c (int): Number of choices in the election.
        num_simulations (int): Number of simulations to perform.
        distribution (str): The type of distribution to use for generating profiles ('cubic', 'box', or 'IANC').
        num_workers (int): The number of parallel workers for simulation.
        output_dir (str): The directory where simulation results will be saved.
        debug (bool, optional): If True, the full preference for each voter will be saved for debugging purposes.

    Returns:
        dict: A dictionary containing the simulation results for all rules.

    Raises:
        ValueError: If an unsupported distribution is provided.
    """
    if distribution == "cubic":
        random_profile = GeneratorProfileCubicUniform(n_v=n_v, n_c=n_c)
    elif distribution == "box":
        random_profile = GeneratorProfileEuclideanBox(
            n_v=n_v, n_c=n_c, box_dimensions=np.ones(2)
        )
    elif distribution == "IANC":
        random_profile = GeneratorProfileIanc(n_v=n_v, n_c=n_c)
    else:
        raise ValueError(f"{distribution} not a supported distribution. Aborting")

    num_distinct_pairs = int(n_c * (n_c - 1) / 2)
    simulation_results = dict()
    if num_workers <= 1:
        # Prepare data structures for running locally
        for rule_name in all_rules:
            simulation_results[f"{rule_name}_winner"] = -np.ones((num_simulations))
            if all_rules[rule_name].is_point_rule:
                simulation_results[rule_name] = np.zeros([num_simulations, n_c])
            else:
                simulation_results[rule_name] = np.zeros(
                    [num_simulations, 2, num_distinct_pairs]
                )
    if debug:
        # the debug data will include the explicit preference for each voter
        simulation_results["debug"] = np.zeros([num_simulations, n_v, n_c])
    if num_workers > 1:
        # parallel implementation
        results = process_map(
            simulate_one_profile,
            [random_profile for i in range(num_simulations)],
            [all_rules for i in range(num_simulations)],
            desc=f"Simulating all rules for n_v={n_v}, n_c={n_c}",
            max_workers=min(num_workers, num_simulations),
            chunksize=1000,
        )
        # stack every results type into one matrix
        for key in results[0]:
            simulation_results[key] = np.vstack(
                [np.expand_dims(res[key], 0) for res in results]
            )
    else:
        # single thread implementation.
        for ind in range(num_simulations):
            profile = random_profile()
            simulation_results["debug"][ind] = profile.preferences_rk
            for rule_name in all_rules:
                (
                    simulation_results[rule_name][
                        ind,
                    ],
                    simulation_results[f"{rule_name}_winner"][ind],
                ) = generate_one_rule_results_for_one_profile(
                    profile, all_rules[rule_name]
                )
    # save results for each rule for this simulation combination.
    for key in simulation_results:
        pkl.dump(
            simulation_results[key],
            open(
                os.path.join(
                    output_dir,
                    f"{key}_{n_v}_{n_c}_sims_{num_simulations}_{distribution}.pkl",
                ),
                "wb",
            ),
        )
    return simulation_results


def simulate_voting(args):
    """
    Perform a comprehensive simulation of voting scenarios for different combinations of voters and choices.

    This function initializes the random seed and then performs simulations for all combinations of the
    number of voters and number of choices defined in the input.
    The output is saved in a separate file for each combination of (num_voters, num_choices)
    The output is the:
    1. Duels matrix for the majority rule (dimensions num_simulations * num_unique_choice_pairs)
    2. The sum of the scores for each choice option for point rules
    (dimensions num_simulations * num_choices)

    Args:
        args (argparse.Namespace): Command-line arguments including distribution, output directory, and
        simulation settings.
    """
    distribution = args.distribution
    num_simulations = args.num_simulations
    num_workers = args.num_workers
    output_dir = args.output_dir
    initialize_random_seeds(int(time.time()))
    # initialize the simulation grid for the number of voters and number of choices
    num_voters = args.num_voters
    if distribution == "IANC":
        max_num_voters_ianc = 15
        num_voters = [num for num in num_voters if num < max_num_voters_ianc]
    num_choices = args.num_choices
    all_rules = initialize_all_voting_rules()
    if args.debug:
        num_choices = [PARAM_PAIR_FOR_DEBUG[1]]
        num_voters = [PARAM_PAIR_FOR_DEBUG[0]]
        args.num_workers = 1

    for n_c in sorted(num_choices, reverse=False):
        for n_v in sorted(num_voters, reverse=False):
            print(f"Simulating n_v = {n_v}, n_c = {n_c}")
            t = time.time()
            simulate_voting_for_specific_nc_nv(
                all_rules=all_rules,
                n_v=n_v,
                n_c=n_c,
                num_simulations=num_simulations,
                distribution=distribution,
                num_workers=num_workers,
                output_dir=output_dir,
                debug=args.debug,
            )
            print(
                f"Simulation took {time.time() - t} seconds ({(time.time() - t)/60} minutes)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--distribution",
        type=str,
        default="cubic",
        help="distribution type to use for the simiulation: cubic / box (euclidean box) / IANC",
    )
    parser.add_argument(
        "-v",
        "--num_voters",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51, 1001, 10001],
        help="a list of the different"
        "number of voters to use in the "
        "grid simulation",
    )
    parser.add_argument(
        "-c",
        "--num_choices",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10],
        help="a list of the different"
        "number of choices to use in the "
        "grid simulation",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=6,
        help="num of workers to use in parallel",
    )
    parser.add_argument(
        "-n",
        "--num_simulations",
        type=int,
        default=1000,
        help="num simulations to perform",
    )
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument(
        "-dbg",
        "--debug",
        action="store_true",
        default=False,
        help="create debug data with the full profile per voter",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        raise ValueError(f"Output dir {args.output_dir} doesn't exist. Aborting")
    simulate_voting(args)
