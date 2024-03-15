# Evaluate voting rules by comparing the preference erosion caused by each of them

## This github repository supplies all the code used to generate the results in the paper
# Balancing Democracy Majoritarianism vs. Expression of Preference Intensity (Nitzan, Nitzan 2023)

## Installation (Linux)
- git clone the repo
- create a virtual environment for the repo
- pip install -r requirements.txt
- Either:
  - Copy the contents of code/profile.py into the similar file installed in the svvamp package (svvamp/preferences/profile.py)
  - Delete from code/rule_loss_simulation_v2.py the AV and IPR rules (lines 71-80)

## Creating simulation results
- All the code exists in the code directory
- From within the repo, run the script code/rule_loss_simulation_v2.py while supplying an output directory (-o). The default parameters are
  - num simulations 1000
  - distribution type cubic
  - Num voters to create simulations for [3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51, 1001, 10001]
  - Num choices to create simulations for [3, 4, 5, 6, 7, 8, 9, 10]
  - Output directory The results directory in the repository

## Evaluating simulation results and creating statistical reports
- From within the repo, run the script code/create_stats.py adjusting the parameters accordingly
- The list of voting rule pairs available for comparison is currently hard coded in the main function
  - MR (Majority rule)
  - BR (Borda rule)
  - PR (Plurality rule)
  - AV (Approval voting)
  - IPR (inverse plurality rule)
- The erosion measures, where x is the option that was not chosen and y is the option that was chosen:
  - B(x)-B(y)/B(x). What proportion of the unchosen option "score" was "ignored". Default
  - B(x)-B(y)/(B(x)+B(y)). What proportion of the total scores involved in the specific choice pair was "ignored"
  - B(x)/B(y) -  the amount of "social injustice", even though x is preferred to y by B(x)/B(x) - it was still not chosen.
- A directory will be created for the results of each pair of voting rules.
- The script will generate csv files with different statistics comparing the erosion between the two rules

## Creating the analytical results and reports
- From within the repo run the script calc_analytical_expectation.py supplying an output directory for the results
- The default configuration
	- Num choices 3
	- Num voters [3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51]
- Larger numbers might cause an out of memory crash. Tune the parameters according to the machine you are using.
- The output of the script will be output files in the same format as the simulation results, but covering all possible preference profile configurations and the expected number of instances for each configuration (calculated using the multinomial distribution coefficients).
- To create the statistical reports for the analytical results, run code/create_stats.py with the argument "-ana" (choosing the relevant input/output directories, and the rules you want to create the report for). This will create similar report directories and CSV files as for the simulation results but for the analytical results.

### Sample results for 100000 simulations are supplied in the "results" directory
### For more details regarding the erosion measures please refer to the article
### For any paper using the code base or mentioning the results, please quote Balancing Democracy Majoritarianism vs. Expression of Preference Intensity (Nitzan and Nitzan 2023)
### Known caveats of the code
- All simulation and analytical results are saved to disk and are read to RAM when creating the statistical reports. This might cause memory issues for large values of voters or
choices
- It should be possible to streamline the simulation and the gathering of data statistics so that there would be no need
to save all intermediate results, but this hasn't been implemented yet.
