from turtle import st
import numpy as np
from scipy.stats import norm

from element.problem_setup import ncs as NCS
from element.problem_setup import na as NA
from element.problem_setup import unit_costs, cs_pfs

from torchrl.envs.utils import ExplorationType


# region: constants for pygad_reliability.py ==================================
# ELE_GA_SEED_FOR_PyGAD = 0
# ELE_GA_POP = 128 #128                            #benchmark = 128       # Population size - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
# ELE_GA_GENS = 256 #256                           #benchmark = 256       # Number of generations - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)

# ELE_GA_LB_BETA = norm.ppf(1-max(cs_pfs))  # 2.5
# ELE_GA_UB_BETA = norm.ppf(1-min(cs_pfs))  # 4.2

# ELE_GA_KEEP_PARENTS = 13                     #benchmark = 13       # 10% of pop=128 / number of parents to keep in the next generation  

# ELE_GA_PARENT_SELECTION = "tournament"       # benchmark:"tournament"
# K_TOURNAMENT=3                               # benchmark = 3

# ELE_GA_CROSSOVER_TYPE = "uniform"            # benchmark = "uniform"     Rationale: cause  genes are continuous β-thresholds; Randomly selects each gene from one of the parents


# ELE_GA_CROSSOVER_PROBABILITY = None
# if ELE_GA_CROSSOVER_TYPE == "uniform" or ELE_GA_CROSSOVER_TYPE == "scattered":
#     """
#     - PyGAD compares each gene in the two parent solutions.
#         - For each gene 'position':
#         - It generates a random number between 0 and 1.
#             - If that number is less than 0.7, the gene is swapped between the parents.
#             - If it's greater than or equal to 0.7, the gene is kept as-is.
#     """
#     ELE_GA_CROSSOVER_PROBABILITY = None      # benchmark =(None means crossover is applied to every mating pair)  - This parameter is used only in 'uniform' crossover or scattered crossover



# MUTATION_TYPE="random"                    # benchmark = random # mutate genes by drawing random numbers (as opposed to e.g. swap/scramble for permutations). Best for continuous genes like your β-thresholds.
#                                           # genes by either replacing them or nudging them — depending on the value of mutation_by_replacement
# MUTATION_BY_REPLACEMENT=False             # benchmark = False    # nudge instead of replace / genes by either replacing them(True) or nudging them(False)
# RANDOM_MUTATION_MIN_VAL=-1.0 #-0.10       # benchmark = -1.0
# RANDOM_MUTATION_MAX_VAL=1.0 #+0.10        # benchmark = 1.0   # small β step (+0.10)
# MUTATION_NUM_GENES = 1                    # benchmark =1 / 1 number of genes to mutate in a solution



# ELE_GA_MAX_COST = unit_costs.max()

# # Initial distribution control (reset-style)
# ELE_GA_RESET_PROB = None
# ELE_GA_DIRICHLET_ALPHA = [0.05594704, 0.16108377, 0.05494736, 0.03863813] # For all states: [0.15481776, 0.07666929, 0.04912562, 0.03946825] #0.5*np.ones(NCS)
# ELE_GA_RANDOM_STATE = 42
# # ELE_GA_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB = np.array([0.0, 0.0, 0.1, 0.8, 0.1])
# # ELE_GA_DIRICHLET_ALPHA = None
# # ELE_GA_RANDOM_STATE = 'off'

# # Inputs for Evaluation part: To compare GA with PPO(evaluation part)
# ELE_GA_HORIZON = 5 #35
# ELE_GA_N_EPISODES_EVAL = 1 # modified to avoid confusion
# ELE_GA_MAX_COST_EVAL = 1.0



# ELE_GA_RESET_PROB_EVAL = None
# ELE_GA_DIRICHLET_ALPHA_EVAL = [0.05594704, 0.16108377, 0.05494736, 0.03863813] # For all states: [0.15481776, 0.07666929, 0.04912562, 0.03946825] #0.5*np.ones(NCS)
# ELE_GA_RANDOM_STATE_EVAL = 42

# # ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB_EVAL = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB_EVAL = np.array([0.0, 0.0, 0.1, 0.8, 0.1])
# # ELE_GA_DIRICHLET_ALPHA_EVAL = None
# # ELE_GA_RANDOM_STATE_EVAL = 'off'

# ELE_GA_INC_STEP_EVAL = True

# ELE_GA_EXPLORE_TYPE_EVAL = ExplorationType.DETERMINISTIC

# endregion ==============================================================


# NEW SET of constants for pygad_reliability_v2.py ==========================
ELE_GA_SEED_FOR_PyGAD = 0
ELE_GA_POP = 512 #512#128 #80                        # Population size - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
ELE_GA_GENS = 1024# 1500 #256 #256 #200              # Number of generations - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
# ELE_GA_LB_BETA, ELE_GA_UB_BETA = 0.0, 8.0          # typical β range (pf ~ 0.5 down to 1e-15)
ELE_GA_LB_BETA = norm.ppf(1-max(cs_pfs))  # 2.5
ELE_GA_UB_BETA = norm.ppf(1-min(cs_pfs))  # 4.2

ELE_GA_KEEP_PARENTS = 2                             # 10% of pop=128 / number of parents to keep in the next generation  

# ELE_GA_PARENT_SELECTION = "sss"                    # steady-state selection
ELE_GA_PARENT_SELECTION = "tournament"               # deafult:parent_selection_type="sss"
K_TOURNAMENT=20 #5 # deafult = 3

# ELE_GA_CROSSOVER_TYPE = "single_point"             # single point means Only one point is used to split and recombine the genes(randolyn selected)
ELE_GA_CROSSOVER_TYPE = "uniform"                    #deafult = crossover_type="single_point"     Rationale: cause  genes are continuous β-thresholds; Randomly selects each gene from one of the parents


ELE_GA_CROSSOVER_PROBABILITY = None
if ELE_GA_CROSSOVER_TYPE == "uniform" or ELE_GA_CROSSOVER_TYPE == "scattered":
    """
    - PyGAD compares each gene in the two parent solutions.
        - For each gene 'position':
        - It generates a random number between 0 and 1.
            - If that number is less than 0.7, the gene is swapped between the parents.
            - If it's greater than or equal to 0.7, the gene is kept as-is.
    """
    ELE_GA_CROSSOVER_PROBABILITY = None       #default = None(None means crossover is applied to every mating pair)  - This parameter is used only in 'uniform' crossover or scattered crossover



MUTATION_TYPE="random"                    # mutate genes by drawing random numbers (as opposed to e.g. swap/scramble for permutations). Best for continuous genes like your β-thresholds.
                                          # genes by either replacing them or nudging them — depending on the value of mutation_by_replacement
MUTATION_BY_REPLACEMENT=False             # deafult = False    # nudge instead of replace / genes by either replacing them(True) or nudging them(False)
RANDOM_MUTATION_MIN_VAL=-0.10             # deafult = -1.0
RANDOM_MUTATION_MAX_VAL=0.10              # deafult = 1.0   # small β step (+0.10)
MUTATION_NUM_GENES = 3                    # 1 number of genes to mutate in a solution



ELE_GA_MAX_COST = unit_costs.max()

# Initial distribution control (reset-style)
ELE_GA_RESET_PROB = None
ELE_GA_DIRICHLET_ALPHA = [0.05594704, 0.16108377, 0.05494736, 0.03863813] #0.5*np.ones(NCS)
ELE_GA_RANDOM_STATE = 42
# ELE_GA_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB = np.array([0.0, 0.0, 0.1, 0.8, 0.1])
# ELE_GA_DIRICHLET_ALPHA = None
# ELE_GA_RANDOM_STATE = 'off'

# Inputs for Evaluation part: To compare GA with PPO(evaluation part)
ELE_GA_HORIZON = 5 #35
ELE_GA_N_EPISODES_EVAL = 10000 # modified to avoid confusion
ELE_GA_MAX_COST_EVAL = 1.0

ELE_GA_RESET_PROB_EVAL = None
ELE_GA_DIRICHLET_ALPHA_EVAL = [0.05594704, 0.16108377, 0.05494736, 0.03863813] # For all states: [0.15481776, 0.07666929, 0.04912562, 0.03946825] #0.5*np.ones(NCS)
ELE_GA_RANDOM_STATE_EVAL = 42

# ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB_EVAL = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB_EVAL = np.array([0.0, 0.0, 0.1, 0.8, 0.1])
# ELE_GA_DIRICHLET_ALPHA_EVAL = None
# ELE_GA_RANDOM_STATE_EVAL = 'off'

ELE_GA_INC_STEP_EVAL = True

ELE_GA_EXPLORE_TYPE_EVAL = ExplorationType.DETERMINISTIC
# endregion ==============================================================

