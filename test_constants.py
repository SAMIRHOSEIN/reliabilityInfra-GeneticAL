from turtle import st
import numpy as np
from scipy.stats import norm

from element.problem_setup import ncs as NCS
from element.problem_setup import na as NA
from element.problem_setup import unit_costs, cs_pfs

from torchrl.envs.utils import ExplorationType


# NEW SET of constants for pygad_reliability_v2.py ==========================
ELE_GA_SEED_FOR_PyGAD = 0

ELE_GA_POP = 512  #512                # Population size - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
ELE_GA_GENS = 500  #1024              # Number of generations - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)

# β bounds inferred from state-based failure probabilities (cs_pfs)
ELE_GA_LB_BETA = norm.ppf(1.0 - max(cs_pfs))  # ~2.5
ELE_GA_UB_BETA = norm.ppf(1.0 - min(cs_pfs))  # ~4.2


# ----------------------------- 1) Elitism and Parent Selection -----------------------------
# Parent survival (elitism-like). number of parents to keep in the next generation  .
ELE_GA_KEEP_PARENTS = 10 #2 # ~8% of pop (512) - It means 512 - 40 = 472 offspring are created per generation and 40 parents are directly carried over to the next generation.



# Parent selection: tournament size controls selection pressure (higher = more exploitative, less diversity)
# ELE_GA_PARENT_SELECTION = "sss"                    # steady-state selection
ELE_GA_PARENT_SELECTION = "tournament"               # deafult:parent_selection_type="sss"
K_TOURNAMENT = 5 # 20 deafult = 3

# ----------------------------- 2) Crossover behavior in PyGAD -----------------------------
# Two distinct concepts:
# (1) Pair-level decision: whether crossover is applied to a selected mating pair.
#     - Controlled by `crossover_probability`.
#     - If crossover_probability = x (0 < x < 1): crossover is applied to a mating pair with probability x.
#     - If crossover_probability = None: crossover is applied to EVERY selected mating pair (100%).
#
# (2) Gene-level inheritance rule: how genes are mixed WHEN crossover occurs.
#     - Controlled by `crossover_type`.
#
# ------------------------------------------------------------------------------------
# 1) crossover_type = "single_point"
#
#    Case A: crossover_probability = x
#    - For each selected mating pair:
#        • With probability x, crossover IS applied.
#        • With probability (1 − x), no crossover occurs (a parent is copied, then mutation may apply).
#    - If crossover is applied:
#        • ONE crossover point is selected uniformly at random.
#        • Genes BEFORE the point are inherited from one parent.
#        • Genes AFTER the point are inherited from the other parent.
#
#    Case B: crossover_probability = None
#    - For EVERY selected mating pair (100%):
#        • Crossover IS applied.
#        • ONE crossover point is selected uniformly at random.
#        • Genes BEFORE the point come from one parent.
#        • Genes AFTER the point come from the other parent.
#
# ------------------------------------------------------------------------------------
#
# 2) crossover_type = "two_points"
#
#    Case A: crossover_probability = x
#    - For each selected mating pair:
#        • With probability x, crossover IS applied.
#        • With probability (1 − x), no crossover occurs.
#    - If crossover is applied:
#        • TWO distinct crossover points are selected uniformly at random.
#        • Genes BETWEEN the two points are swapped between parents.
#        • Genes OUTSIDE the two points are inherited from the original parent.
#
#    Case B: crossover_probability = None
#    - For EVERY selected mating pair (100%):
#        • Crossover IS applied.
#        • TWO distinct crossover points are selected uniformly at random.
#        • Genes BETWEEN the two points are swapped.
#        • Genes OUTSIDE the two points remain with the original parent.
#
# ------------------------------------------------------------------------------------

# 3) crossover_type = "uniform"
#
#    Case A: crossover_probability = x
#    - For each selected mating pair:
#        • With probability x, crossover IS applied.
#        • With probability (1 − x), no crossover occurs.
#    - If crossover is applied:
#        • For EACH gene position independently:
#            – The offspring inherits that gene from either parent
#              with approximately equal probability (~50% each).
#
#    Case B: crossover_probability = None
#    - For EVERY selected mating pair (100%):
#        • Crossover IS applied.
#        • For EACH gene position independently:
#            – The offspring inherits that gene from either parent
#              with approximately equal probability (~50% each).
#
# ------------------------------------------------------------------------------------

# 4) crossover_type = "scattered"
#
#    Case A: crossover_probability = x
#    - For each selected mating pair:
#        • With probability x, crossover IS applied.
#        • With probability (1 − x), no crossover occurs.
#    - If crossover is applied:
#        • A random binary mask of chromosome length is generated.
#        • For each gene position:
#            – The mask determines whether the gene is inherited from parent A or parent B.
#
#    Case B: crossover_probability = None
#    - For EVERY selected mating pair (100%):
#        • Crossover IS applied.
#        • A random binary mask is generated.
#        • For each gene position:
#            – The mask determines whether the gene is inherited from parent A or parent B.
#
# ====================================================================================
# ELE_GA_CROSSOVER_TYPE = "single_point"             # single point means Only one point is used to split and recombine the genes(randolyn selected)
ELE_GA_CROSSOVER_TYPE = "uniform"                    #deafult = crossover_type="single_point"     Rationale: cause  genes are continuous β-thresholds; Randomly selects each gene from one of the parents

ELE_GA_CROSSOVER_PROBABILITY = None  #  if None => 100% of mating pairs crossover

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


# ----------------------------- 3) Mutation (random + nudge) -----------------------------
# Mutation is applied AFTER crossover to each offspring.
#
# mutation_type = "random":
#   - Suitable for continuous-valued genes (e.g., β-thresholds).
#
# mutation_by_replacement:
#   - False  => NUDGE mutation (local search):
#       gene_new = gene_old + δ
#       where δ ~ Uniform(RANDOM_MUTATION_MIN_VAL, RANDOM_MUTATION_MAX_VAL)
#       Note: RANDOM_MUTATION_MIN_VAL / RANDOM_MUTATION_MAX_VAL: Used ONLY for NUDGE mutation (mutation_by_replacement=False).
#
#   - True   => REPLACEMENT mutation (global jump):
#       The mutated gene is replaced by a new value sampled from the gene's allowed space.
#       If gene_space is set to [ELE_GA_LB_BETA, ELE_GA_UB_BETA], then replacement is effectively:
#           gene_new ~ Uniform(ELE_GA_LB_BETA, ELE_GA_UB_BETA)
#
# mutation_num_genes:
#   - Number of genes (out of 4) selected uniformly at random (without replacement)
#     to be mutated in each offspring.
#   - For 4 genes:
#       * mutation_num_genes = 1  -> mild mutation (fine-tuning)
#       * mutation_num_genes = 2  -> moderate mutation (balanced)
#       * mutation_num_genes = 3  -> aggressive mutation (75% of chromosome perturbed)
#

MUTATION_TYPE="random"                    # mutate genes by drawing random numbers. Best for continuous genes like your β-thresholds.
MUTATION_BY_REPLACEMENT=False             # deafult = False    
RANDOM_MUTATION_MIN_VAL=-0.10             # deafult = -1.0
RANDOM_MUTATION_MAX_VAL=0.10              # deafult = 1.0   # small β step (+0.10)
MUTATION_NUM_GENES = 2   #3                  # 1 number of genes to mutate in a solution



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