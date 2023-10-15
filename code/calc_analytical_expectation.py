import argparse
import os
import pickle as pkl
import time
from typing import List

import numpy as np
import svvamp
from sympy import ntheory
from sympy.utilities.iterables import multiset_permutations

from rule_loss_simulation_v2 import (
    initialize_all_voting_rules,
    generate_one_rule_results_for_one_profile,
)


def create_all_possible_preference_profiles(n_c: int):
    all_profiles = multiset_permutations(list(range(n_c)))
    res = np.zeros((np.math.factorial(n_c), n_c))
    for ind, profile in enumerate(all_profiles):
        res[ind, :] = profile
    return res


class CalcAnalyticalErosionExpection:
    """
    Calculate erosion expectation over all pairs for a specific number of choices
    and number of voters
    Class parameters
        n_c - num choices
        n_v - num voters
        num_distinct_pairs - int(n_c * (n_c - 1) / 2)
        all_rules - all possible implemented voting rules
        all_possible_preference_profiles - all possible unique rankings that one
        voter could have for ranking num_choices options
        multinomial_calculator - calculates the multinomial coefficient for each one
        of the unique combinations that num_voters could rank num_choices options
    """

    def __init__(self, n_c: int, n_v: int) -> None:
        self.n_c = n_c
        self.n_v = n_v
        self.num_distinct_pairs = int(n_c * (n_c - 1) / 2)
        self.all_rules = initialize_all_voting_rules()
        self.all_possible_preference_profiles = create_all_possible_preference_profiles(
            n_c
        )
        self.multinomial_calculator = ntheory.multinomial_coefficients(
            self.all_possible_preference_profiles.shape[0], n_v - 1
        )

    def create_profile_calculator_for_one_possible_profile_combination(self, profile_comb: np.array):
        """
        Given an index of the profile combination choice in the multinomial_calculator
        Create the svvamp profile for that profile. This will later be used to create
        the data to be saved per rule.
        """
        preferences_borda_rk = [
            self.all_possible_preference_profiles[ind, :]
            for ind in range(len(profile_comb))
            for x in range(profile_comb[ind])
        ]
        preferences_borda_rk.append(self.all_possible_preference_profiles[0, :])
        preferences_borda_rk = np.vstack(preferences_borda_rk)
        return svvamp.Profile(preferences_ut=preferences_borda_rk)

    def calc_weight_and_erosion_for_all_profile_combinations(self, output_dir: str):
        analytical_results = dict()
        num_unique_profile_combinations = len(self.multinomial_calculator)
        # initialize output data structures
        for rule_name in self.all_rules:
            analytical_results["profile_weight"] = -np.ones(
                [num_unique_profile_combinations]
            )
            analytical_results[f"{rule_name}_winner"] = -np.ones(
                (num_unique_profile_combinations)
            )
            if self.all_rules[rule_name].is_point_rule:
                analytical_results[rule_name] = np.zeros(
                    [num_unique_profile_combinations, self.n_c]
                )
            else:
                analytical_results[rule_name] = np.zeros(
                    [num_unique_profile_combinations, 2, self.num_distinct_pairs]
                )

        # fill data structures with all possible profiles results
        for ind, profile_comb in enumerate(self.multinomial_calculator):
            analytical_results["profile_weight"][ind] = self.multinomial_calculator[
                profile_comb
            ]
            profiles_results = self.create_profile_calculator_for_one_possible_profile_combination(
                profile_comb
            )
            for rule_name in self.all_rules:
                (
                    analytical_results[rule_name][
                        ind,
                    ],
                    analytical_results[f"{rule_name}_winner"][ind],
                ) = generate_one_rule_results_for_one_profile(
                    profiles_results, self.all_rules[rule_name]
                )
        # Save all data structures
        for key in analytical_results:
            pkl.dump(
                analytical_results[key],
                open(
                    os.path.join(
                        output_dir, f"{key}_{self.n_v}_{self.n_c}_analytical_res.pkl"
                    ),
                    "wb",
                ),
            )


def calc_analytical_expectation_for_specific_nc_nv(n_v: int, n_c: int, output_dir: str):
    calc = CalcAnalyticalErosionExpection(n_c, n_v)
    calc.calc_weight_and_erosion_for_all_profile_combinations(output_dir=output_dir)


def calc_analytical_expectation(num_voters: List[int], num_choices: List[int], output_dir: str):
    """
    Create data for calculating the analytical expectations of the erosion
    between different rules. The data includes the erosion for all possible unique
    profile combinations and a weight signifying how many times would that unique
    profile appear in expectation (the multinomial coefficient of the profile).
    The actual expectation reports can be created using the create_stats.py script.

    Results are stored in a speparate file for each (num_voters, num_choices) combination

    The number of voters and number of choices were limited to the ones that could
    run on my personal i7, 32GB ram computer.
    """
    for n_v in sorted(num_voters, reverse=False):
        for n_c in sorted(num_choices, reverse=False):
            print(f"Analytical calculation n_v = {n_v}, n_c = {n_c}")
            t = time.time()
            calc_analytical_expectation_for_specific_nc_nv(
                n_v=n_v, n_c=n_c, output_dir=output_dir
            )
            print(
                f"Analytical calculation took {time.time() - t} seconds ({(time.time() - t)/60} minutes)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--num_voters",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51],
        help="a list of the different"
        "number of voters to use in the "
        "grid simulation",
    )
    parser.add_argument(
        "-c",
        "--num_choices",
        type=int,
        nargs="+",
        default=[3],
        help="a list of the different"
        "number of choices to use in the "
        "grid simulation",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    calc_analytical_expectation(args.num_voters, args.num_choices, args.output_dir)
