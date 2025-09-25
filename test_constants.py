import numpy as np

from element.problem_setup import ncs as NCS
from element.problem_setup import na as NA
from element.problem_setup import unit_costs

from torchrl.envs.utils import ExplorationType

# region: constants for ele_exp_const.py =====================================
ELE_CONST_HORIZON = 1
# ELE_CONST_N_HORIZON = 1
ELE_CONST_N_EPISODES = 1 # modified to avoid confusion
ELE_CONST_MAX_COST = 1.0

# ELE_CONST_RESET_PROB = None
# ELE_CONST_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_CONST_RANDOM_STATE = 24
ELE_CONST_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
ELE_CONST_DIRICHLET_ALPHA = None
ELE_CONST_RANDOM_STATE = 'off'

ELE_CONST_ACTION = 1

ELE_CONST_EXPLORE_TYPE = ExplorationType.RANDOM
# endregion ==============================================================


# region: constants for ele_exp_custom.py ====================================
ELE_CUSTOM_HORIZON = 1
# ELE_CUSTOM_N_HORIZON = 1
ELE_CUSTOM_N_EPISODES = 1 # modified to avoid confusion
ELE_CUSTOM_MAX_COST = 1.0

# ELE_CUSTORM_RESET_PROB = None
# ELE_CUSTORM_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_CUSTORM_RANDOM_STATE = 24
ELE_CUSTOM_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
ELE_CUSTOM_DIRICHLET_ALPHA = None
ELE_CUSTOM_RANDOM_STATE = 'off'


ELE_CUSTOM_EXPLORE_TYPE = ExplorationType.RANDOM
# endregion ==============================================================


# region: constants for ele_ppo_training.py ==================================
# env parameters
ELE_PPO_HORIZON = 5 #75
ELE_PPO_INC_STEP = True
ELE_PPO_MAX_COST = unit_costs.max()

# ELE_PPO_RESET_PROB = None
# ELE_PPO_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_PPO_RANDOM_STATE = 42
ELE_PPO_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
ELE_PPO_DIRICHLET_ALPHA = None
ELE_PPO_RANDOM_STATE = 'off'

# network parameters
ELE_PPO_TORCH_SEED = 0
if ELE_PPO_INC_STEP:
    ELE_PPO_INPUT_DIM = NCS + 1
else:
    ELE_PPO_INPUT_DIM = NCS
ELE_PPO_ACTOR_CELLS = 32
ELE_PPO_ACTOR_LAYERS = 2
ELE_PPO_VALUE_CELLS = 32
ELE_PPO_VALUE_LAYERS = 2

ELE_PPO_OUTPUT_DIM = NA

# GAE parameters
# gamma has to be 1 to avoid double counting gamma in the env
# lmbda=0 is equivalent to using TD0
# lmbda=1 is equivalent to using TD1
# so lmbda should be between 0 and 1
ELE_PPO_GAE_GAMMA = 1.0
ELE_PPO_GAE_LAMBDA = 0.95
ELE_PPO_AVERAGE_GAE = True

# PPO loss parameters
ELE_PPO_ENTROPY_EPS = 0.01
ELE_PPO_CLIP_EPSILON = (1e-3)
ELE_PPO_CRITIC_COEF = 1.0

# collector parameters
ELE_PPO_EPISODES_PER_BATCH = 32     # how many episodes we collect per iteration 
ELE_PPO_NUM_ITERATIONS = 1024       # how many iteration: collect→optimize cycles we run in total
ELE_PPO_FRAMES_PER_BATCH = ELE_PPO_HORIZON*ELE_PPO_EPISODES_PER_BATCH
ELE_PPO_TOTAL_FRAMES = ELE_PPO_FRAMES_PER_BATCH*ELE_PPO_NUM_ITERATIONS
ELE_PPO_SPLIT_TRAJS = False

# training parameters
ELE_PPO_TRAINING_EPOCHS = 50
ELE_PPO_SUB_BATCH_SIZE = ELE_PPO_HORIZON*32 # actually we consider one one mini-batch
ELE_PPO_MAX_GRAD_NORM = 1.0
ELE_PPO_LR = 1e-3
ELE_PPO_LR_MIN = 1e-5    # lr reduced to lr_min with total_frames // frames_per_batch
ELE_PPO_EVAL_FREQ = 1
ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================


# region: constants for ele_exp_actor.py ====================================
# ELE_ACTOR_VERSION = '20250505-192030'   #David's model
# ELE_ACTOR_VERSION = '20250910-202015' # my model with 5 horizon
# ELE_ACTOR_VERSION = '20250917-102249' # my model with 75 horizon
# ELE_ACTOR_VERSION = '20250924-173308' # my model with 1 horizon with dirichlet alpha 0.5
# ELE_ACTOR_VERSION = '20250924-183258' # my model with 1 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250924-184355' # my model with 5 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250924-190413' # my model with 10 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250925-100427'   # my model with 1 horizon with reset prob [1,0,0,0,0]
ELE_ACTOR_VERSION = '20250925-101620'   # my model with 5 horizon with reset prob [1,0,0,0,0]


ELE_ACTOR_HORIZON = 5 #75
# ELE_ACTOR_N_HORIZON = 1
ELE_ACTOR_N_EPISODES = 1 # modified to avoid confusion
ELE_ACTOR_MAX_COST = 1.0

# ELE_DP_RESET_PROB = None
# ELE_DP_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_DP_RANDOM_STATE = 42
ELE_ACTOR_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_ACTOR_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# ELE_ACTOR_RESET_PROB = np.array([0.0, 0.8, 0.2, 0.0, 0.0])

ELE_ACTOR_DIRICHLET_ALPHA = None
ELE_ACTOR_RANDOM_STATE = 'off'


ELE_ACTOR_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================



# region: constants for DPvsPPO.py ==================================
ELE_DP_HORIZON = 5 #75
ELE_DP_N_EPISODES = 1 # In DP we always consider 1 episode
ELE_DP_MAX_COST = 1.0


ELE_DP_INC_STEP = True
# if ELE_DP_INC_STEP:
#     ELE_DP_INPUT_DIM = NCS + 1
# else:
#     ELE_DP_INPUT_DIM = NCS


# ELE_DP_RESET_PROB = None
# ELE_DP_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_DP_RANDOM_STATE = 42
ELE_DP_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_DP_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# ELE_DP_RESET_PROB = np.array([0.0, 0.8, 0.2, 0.0, 0.0])

ELE_DP_DIRICHLET_ALPHA = None
ELE_DP_RANDOM_STATE = 'off'


ELE_DP_EXPLORE_TYPE = ExplorationType.DETERMINISTIC
# endregion ==============================================================

# region: constants for pygad_reliability.py ==================================
ELE_GA_SEED_FOR_PyGAD = 0
ELE_GA_POP = 128 #80                                             # Population size
ELE_GA_GENS = 256 #200                                           # Number of generations
ELE_GA_LB_BETA, ELE_GA_UB_BETA = 0.0, 8.0                   # typical β range (pf ~ 0.5 down to 1e-15)


ELE_GA_MUTATION_PERCENT_GENES = 50                          # mutate 50% of genes per solution
# ELE_GA_MUTATION_PERCENT_GENES = 20                         #(50% is quite aggressive for 4 decision thresholds; 15–25% works better in my experience.)

ELE_GA_CROSSOVER_TYPE = "single_point"                      # single point means Only one point is used to split and recombine the genes
# ELE_GA_CROSSOVER_TYPE = "uniform"                         # Rationale: your genes are continuous β-thresholds; uniform crossover avoids positional bias of single-point.

ELE_GA_PARENT_SELECTION = "sss"                             # steady-state selection
# ELE_GA_PARENT_SELECTION = "tournament"                    # (PyGAD defaults K_tournament=3; this typically improves pressure without killing diversity.)

ELE_GA_KEEP_PARENTS = 2                                     # number of parents to keep in the next generation    
ELE_GA_MAX_COST = unit_costs.max()

# Initial distribution control (reset-style)
# ELE_GA_RESET_PROB = None
# ELE_GA_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_GA_RANDOM_STATE = 42
ELE_GA_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
ELE_GA_DIRICHLET_ALPHA = None
ELE_GA_RANDOM_STATE = 'off'


# Inputs for Evaluation part: To compare GA with PPO(evaluation part)
ELE_GA_HORIZON = 5 #75
ELE_GA_N_EPISODES_EVAL = 1 # modified to avoid confusion
ELE_GA_MAX_COST_EVAL = 1.0


# ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_GA_DIRICHLET_ALPHA_EVAL = None
# ELE_GA_RANDOM_STATE_EVAL = 'off'
ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_GA_RESET_PROB_EVAL = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# ELE_GA_RESET_PROB_EVAL = np.array([0.0, 0.8, 0.2, 0.0, 0.0])

ELE_GA_DIRICHLET_ALPHA_EVAL = None
ELE_GA_RANDOM_STATE_EVAL = 'off'


ELE_GA_INC_STEP_EVAL = True

ELE_GA_EXPLORE_TYPE_EVAL = ExplorationType.DETERMINISTIC
# endregion ==============================================================