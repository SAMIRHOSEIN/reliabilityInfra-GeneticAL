#%%
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Final version of the PyGAD script
#-------------------------------------------------------------------------------------------------------------------------------------------------
import json 
import numpy as np
from scipy.stats import norm
import pygad
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
import torch
from collections import defaultdict
from tqdm import tqdm
from torchrl.envs.utils import set_exploration_type
from torchrl_bridge import create_element_env
from element.problem_setup import ncs, cs_pfs
import test_constants
import sys
start_time = time.time()
# -------------------------- Load problem setup --------------------------
from element.problem_setup import (
    ncs, na, gamma,
    action_model, unit_costs, cs_pfs, failure_cost
)

from element.utility_func import cost_util
# -------------------------- GA & model knobs (ALL HERE) -----------------

#load constants
SEED = test_constants.ELE_GA_SEED_FOR_PyGAD
NUM_GENERATIONS = test_constants.ELE_GA_GENS
POPULATION_SIZE = test_constants.ELE_GA_POP
NUM_PARENTS_MATING = max(2, POPULATION_SIZE // 2) # ensure at least 2 parents
MUTATION_NUM_GENES = test_constants.MUTATION_NUM_GENES
CROSSOVER_TYPE = test_constants.ELE_GA_CROSSOVER_TYPE
PARENT_SELECTION = test_constants.ELE_GA_PARENT_SELECTION
KEEP_PARENTS = test_constants.ELE_GA_KEEP_PARENTS
ELE_GA_LB_BETA = test_constants.ELE_GA_LB_BETA
ELE_GA_UB_BETA = test_constants.ELE_GA_UB_BETA
ELE_GA_MAX_COST = test_constants.ELE_GA_MAX_COST
ELE_GA_RANDOM_STATE = test_constants.ELE_GA_RANDOM_STATE
ELE_GA_RESET_PROB = test_constants.ELE_GA_RESET_PROB
ELE_GA_DIRICHLET_ALPHA = test_constants.ELE_GA_DIRICHLET_ALPHA
HORIZON = test_constants.ELE_GA_HORIZON
CROSSOVER_PROBABILITY = test_constants.ELE_GA_CROSSOVER_PROBABILITY
K_TOURNAMENT = test_constants.K_TOURNAMENT
MUTATION_TYPE = test_constants.MUTATION_TYPE
mutation_by_replacement = test_constants.MUTATION_BY_REPLACEMENT
RANDOM_MUTATION_MIN_VAL = test_constants.RANDOM_MUTATION_MIN_VAL
RANDOM_MUTATION_MAX_VAL = test_constants.RANDOM_MUTATION_MAX_VAL


print(f"GA inputs:\n"
      f"pop={POPULATION_SIZE}\n"
      f"gens={NUM_GENERATIONS}\n"
      f"parent_sel='{PARENT_SELECTION}'\n"
      f"cross='{CROSSOVER_TYPE}'\n"
      f"crossover_probability={CROSSOVER_PROBABILITY}\n"
      f"num_parents_mating={NUM_PARENTS_MATING}\n"  # Number of solutions to be selected as parents
      f"mutation_type={MUTATION_TYPE}\n"
      f"mut(num genes)={MUTATION_NUM_GENES}\n"
      f"keep_parents={KEEP_PARENTS}\n"
      f"horizon={HORIZON}\n"
      f"ELE_GA_RESET_PROB={ELE_GA_RESET_PROB}\n"
      f"ELE_GA_DIRICHLET_ALPHA={ELE_GA_DIRICHLET_ALPHA}\n"
      f"ELE_GA_LowerBound_BETA={ELE_GA_LB_BETA}\n"
      f"ELE_GA_UpperBound_BETA={ELE_GA_UB_BETA}\n")

# -------------------------- Prepare arrays ------------------------------
# Keep the same shapes used in rl_env.step():
P_actions = np.asarray(action_model, float)   # (na, ncs, ncs)
Unit_costs = np.asarray(unit_costs, float)    # (na, ncs)  (we'll index Unit_costs[action] @ state)
pf_array   = np.clip(np.asarray(cs_pfs, float), 1e-12, 1 - 1e-12)  # (ncs,)

# # -------------------------- Build STATE0 once (reset) -------------------
# if ELE_GA_RANDOM_STATE == 'off':
#     assert ELE_GA_RESET_PROB is not None, "ELE_GA_RESET_PROB must be set when RANDOM_STATE is 'off'."
#     STATE0 = np.asarray(ELE_GA_RESET_PROB, float).ravel()
#     assert STATE0.shape[0] == ncs and np.isclose(STATE0.sum(), 1.0), "ELE_GA_RESET_PROB must sum to 1."
# else:
#     assert ELE_GA_DIRICHLET_ALPHA is not None, "ELE_GA_DIRICHLET_ALPHA must be set when RANDOM_STATE is used."
#     rng = np.random.default_rng(ELE_GA_RANDOM_STATE)
#     STATE0 = rng.dirichlet(np.asarray(ELE_GA_DIRICHLET_ALPHA, float).ravel())

# -------------------------- Policy & rollout ----------------------------
def reliability_based_action(beta_t, betas_desc):
    """betas_desc is sorted DESC: [β1 ≥ β2 ≥ β3 ≥ β4]."""
    beta_ac1, beta_ac2, beta_ac3, beta_ac4 = betas_desc
    # Original version:
    # if beta_t <= beta_ac4: return 4
    # if beta_t <= beta_ac3: return 3
    # if beta_t <= beta_ac2: return 2
    # if beta_t <= beta_ac1: return 1
    # return 0 # do nothing
    # modified after discussion with David
    if beta_t > beta_ac1:
        action = 0  # do nothing
    elif beta_ac2 <= beta_t < beta_ac1:
        action = 1  # maintenance
    elif beta_ac3 <= beta_t < beta_ac2:
        action = 2  # repair
    elif beta_ac4 <= beta_t < beta_ac3:
        action = 3  # rehabilitation
    elif beta_t <= beta_ac4:
        action = 4  # replacement
    else:
        raise ValueError(f"Unexpected beta_t={beta_t} with thresholds={betas_desc}")
    return action





def _repair_thresholds(betas, low=None, high=None):
    """Clip -> sort high to low -> enforce strict decrease (robust)."""
    x = np.asarray(betas, dtype=float)

    if low is not None or high is not None:
        x = np.clip(x, low, high) # keep inside the defined bounds

    x = np.sort(x)[::-1] # enforce β1>β2>β3>β4

    eps = 1e-6
    for i in range(len(x) - 1):
        if x[i] <= x[i + 1]:
            x[i + 1] = x[i] - eps

    # if strictness pushed below low, re-clip + re-enforce from the end
    if low is not None:
        x[-1] = max(x[-1], low)
        for i in range(len(x) - 2, -1, -1):
            x[i] = max(x[i], x[i + 1] + eps)

    return x




# def rollout_betas(betas_raw):
#     """
#     Simulate one episode over the finite horizon using β-thresholds.
#     Returns:
#       exp_dis_cost (float): expected discounted cost we minimize
#       logs (dict): actions, pf, beta, reward series for inspection
#     """
#     # betas_raw = np.asarray(betas_raw, float)
#     # betas_raw = np.clip(betas_raw, ELE_GA_LB_BETA, ELE_GA_UB_BETA) #clip due to numerical stability and if beta was very large/small, we will face issue due to np.cdf
#     # betas_desc = np.sort(betas_raw)[::-1]

#     betas_desc = _repair_thresholds(betas_raw, low=ELE_GA_LB_BETA, high=ELE_GA_UB_BETA)

#     state = STATE0.copy()   # shape (ncs,), initial state distribution = STATE0 

#     H = int(HORIZON)
#     exp_dis_cost = 0.0
#     actions, pf_series, beta_series, reward_series = [], [], [], []

#     for t in range(H):
#       # prob of failure and reliability index at time t
#       pf_t= float(np.clip(pf_array @ state, 1e-12, 1-1e-12))    #clip due to numerical stability(norm.ppf(1) and norm.ppf(0) are inf) and keep the serch practical.
#       beta_t = float(norm.ppf(1.0 - pf_t))

#       # choose action with thresholds
#       action = reliability_based_action(beta_t, betas_desc)

#       # cost
#       dir_cost  = float(Unit_costs[int(action)] @ state)
#       fail_risk = float((pf_array @ state) * float(failure_cost))
#       cost      = dir_cost + fail_risk

#       # reward(for comparison only)
#       discount_factor = float(gamma) ** t
#       reward = discount_factor * cost_util(cost, min_cost=None, max_cost=ELE_GA_MAX_COST)

#       # objective func
#       exp_dis_cost += discount_factor * cost

#       #log
#       actions.append(int(action))
#       pf_series.append(pf_t)
#       beta_series.append(beta_t)
#       reward_series.append(float(reward))

#       # state transition
#       state = P_actions[int(action)].T @ state
#       state = state / state.sum()  # normalize to fix any rounding

#     logs = dict(
#         betas_desc=betas_desc.tolist(),
#         actions=actions,
#         pf=pf_series,
#         beta=beta_series,
#         reward=reward_series,
#         exp_dis_cost=float(exp_dis_cost)
#     )

#     return exp_dis_cost, logs






# -------------------------- Fixed initial state per population slot  --------------------------
if ELE_GA_RANDOM_STATE == 'off':
    raise ValueError(
        "You asked for different initial states per chromosome, "
        "but ELE_GA_RANDOM_STATE='off' implies a fixed reset_prob. "
        "Set ELE_GA_RANDOM_STATE to an int seed and provide ELE_GA_DIRICHLET_ALPHA."
    )

assert ELE_GA_DIRICHLET_ALPHA is not None, "Need ELE_GA_DIRICHLET_ALPHA to sample initial states."
alpha0 = np.asarray(ELE_GA_DIRICHLET_ALPHA, float).ravel()

rng_init = np.random.default_rng(ELE_GA_RANDOM_STATE)  # reproducible
STATE0_POOL = np.vstack([rng_init.dirichlet(alpha0) for _ in range(POPULATION_SIZE)]).astype(np.float32)
# STATE0_POOL[i] is the fixed initial state for solution_idx=i across all generations



def rollout_betas(betas_raw, state0=None):
    """
    Simulate one episode over the finite horizon using β-thresholds.
    Returns:
    exp_dis_cost: discounted expected cost under state-distribution propagation, conditioned on a given initial distribution state0.
      logs (dict): actions, pf, beta, reward series for inspection
    """
        
    betas_desc = _repair_thresholds(betas_raw, low=ELE_GA_LB_BETA, high=ELE_GA_UB_BETA)


    state = np.asarray(state0, dtype=float).copy()

    H = int(HORIZON)
    exp_dis_cost = 0.0
    actions, pf_series, beta_series, reward_series = [], [], [], []

    for t in range(H):
        # prob of failure and reliability index at time t
        pf_t = float(np.clip(pf_array @ state, 1e-12, 1 - 1e-12))
        beta_t = float(norm.ppf(1.0 - pf_t))

        # choose action with thresholds
        action = reliability_based_action(beta_t, betas_desc)

        # cost
        dir_cost  = float(Unit_costs[int(action)] @ state)
        fail_risk = float((pf_array @ state) * float(failure_cost))
        cost      = dir_cost + fail_risk

        # reward(for comparison only)
        discount_factor = float(gamma) ** t
        reward = discount_factor * cost_util(cost, min_cost=None, max_cost=ELE_GA_MAX_COST)

        # objective func
        exp_dis_cost += discount_factor * cost
        
        #log
        actions.append(int(action))
        pf_series.append(pf_t)
        beta_series.append(beta_t)
        reward_series.append(float(reward))

        # state transition
        state = P_actions[int(action)].T @ state
        state = state / state.sum() # normalize to fix any rounding

    logs = dict(
        betas_desc=betas_desc.tolist(),
        actions=actions,
        pf=pf_series,
        beta=beta_series,
        reward=reward_series,
        exp_dis_cost=float(exp_dis_cost)
    )
    return exp_dis_cost, logs



















# # -------------------------- PyGAD fitness-------------
# #must include all three parameters because PyGAD will pass them,
# def fitness_func(ga_instance, solution, solution_idx):
#     exp_dis_cost, _ = rollout_betas(solution)
#     return float(-exp_dis_cost)  # PyGAD maximizes fitness; we minimize cost.






def fitness_func(ga_instance, solution, solution_idx): # solution_idx is the index of solution in population
    state0 = STATE0_POOL[solution_idx]  # fixed per slot, reproducible across generations
    exp_dis_cost, _ = rollout_betas(solution, state0=state0)
    return float(-exp_dis_cost)






# the following line is for all chormosome in all generation has 32 fix inital states
# EVAL_PANEL_SIZE = 32  # 16–64 is typical; tradeoff speed vs stability
# rng_panel = np.random.default_rng(SEED)
# EVAL_PANEL_IDX = rng_panel.choice(POPULATION_SIZE, size=EVAL_PANEL_SIZE, replace=False)


# def fitness_func(ga_instance, solution, solution_idx):
#     costs = []
#     for idx in EVAL_PANEL_IDX:
#         c, _ = rollout_betas(solution, state0=STATE0_POOL[idx])
#         costs.append(c)
#     return float(-np.mean(costs))







# -------------------------- Run GA --------------------------------------
assert na == 5, "reliability_based_action currently assumes 5 actions (4 thresholds + do-nothing)"
na_exc_zero = na - 1  # num of actions excluding 'do nothing' (action 0)
gene_space = [{'low': ELE_GA_LB_BETA, 'high': ELE_GA_UB_BETA} for _ in range(na_exc_zero)]

# To plot learning curve, we will record the best fitness per generation and also 
# we need to plot Mean/median shows whether the whole population is improving or only one 'lucky superstar' is improving
fitness_history = []
mean_fitness_history = []
median_fitness_history = []


# An optional parameter named on_generation is supported which allows the user to call a function (with a single parameter) after each generation
# Ref:https://pygad.readthedocs.io/en/latest/pygad.html#the-on-generation-parameter 
# def on_gen(ga_instance):
#     """Record the best fitness after each generation."""
#     _, best_fitness, _ = ga_instance.best_solution()
#     fitness_history.append(best_fitness)



def on_gen(ga_instance):
    """Track best/mean/median fitness per generation to diagnose convergence and diversity."""
    pop_fitness = np.asarray(ga_instance.last_generation_fitness, dtype=float)

    # Best of current generation:
    fitness_history.append(float(np.max(pop_fitness)))

    # Central tendency of population:
    mean_fitness_history.append(float(np.mean(pop_fitness)))
    median_fitness_history.append(float(np.median(pop_fitness)))












# # good result(I keep the following as benchmark)
# ga = pygad.GA(
#     num_generations=NUM_GENERATIONS,
#     num_parents_mating=NUM_PARENTS_MATING,
#     fitness_func=fitness_func,
#     sol_per_pop=POPULATION_SIZE,
#     num_genes=na_exc_zero,
#     gene_space=gene_space,
#     init_range_low=ELE_GA_LB_BETA,
#     init_range_high=ELE_GA_UB_BETA,
#     parent_selection_type=PARENT_SELECTION,   # parent selection
#     # K_tournament=K_TOURNAMENT,                # parent selection           
#     crossover_type=CROSSOVER_TYPE,                 # crossover
#     # crossover_probability=CROSSOVER_PROBABILITY,    # crossover
#     mutation_type=MUTATION_TYPE,                                # mutation
#     # mutation_by_replacement=mutation_by_replacement,            # mutation
#     mutation_num_genes=MUTATION_NUM_GENES,             # mutation
#     # random_mutation_min_val=RANDOM_MUTATION_MIN_VAL,            # mutation
#     # random_mutation_max_val=RANDOM_MUTATION_MAX_VAL,            # mutation
#     keep_parents=KEEP_PARENTS,
#     random_seed=SEED,
#     on_generation=on_gen
# )

# the following is the same good result(benchmark) as above but with all deafult values for commented parameters above
ga = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING, #Number of solutions to be selected as parents
    fitness_func=fitness_func,
    sol_per_pop=POPULATION_SIZE,
    num_genes=na_exc_zero,
    gene_space=gene_space,
    init_range_low=ELE_GA_LB_BETA,
    init_range_high=ELE_GA_UB_BETA,
    parent_selection_type=PARENT_SELECTION,   # parent selection
    K_tournament=K_TOURNAMENT,                # parent selection           
    crossover_type=CROSSOVER_TYPE,                 # crossover
    crossover_probability=CROSSOVER_PROBABILITY,    # crossover
    mutation_type=MUTATION_TYPE,                                # mutation
    mutation_by_replacement=mutation_by_replacement,            # mutation
    mutation_num_genes=MUTATION_NUM_GENES,             # mutation
    random_mutation_min_val=RANDOM_MUTATION_MIN_VAL,            # mutation
    random_mutation_max_val=RANDOM_MUTATION_MAX_VAL,            # mutation
    keep_parents=KEEP_PARENTS,
    random_seed=SEED,
    on_generation=on_gen
)


print("PyGAD β-threshold optimization (minimize discounted expected cost)")
ga.run()


# -------------------------- Plot learning curve -------------------------
if fitness_history:
    generations = np.arange(1, len(fitness_history) + 1)
    best_cost_per_gen = -np.asarray(fitness_history, dtype=float)

    plt.figure(figsize=(8, 4))
    plt.plot(generations, best_cost_per_gen, marker='o', linewidth=1.5)
    plt.title("PyGAD Learning Curve")
    plt.xlabel("Generation")
    plt.ylabel("Best expected discounted cost")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    initial_best_cost = float(best_cost_per_gen[0])
    final_best_cost = float(best_cost_per_gen.min())
    absolute_improvement = initial_best_cost - final_best_cost
    relative_improvement = (
        (absolute_improvement / initial_best_cost) * 100.0
        if initial_best_cost != 0.0 else 0.0
    )

    print(f"Initial best cost: {initial_best_cost:.6f}")
    print(f"Final best cost:   {final_best_cost:.6f}")
    print(f"Absolute improvement: {absolute_improvement:.6f}")
    if relative_improvement:
        print(f"Relative improvement: {relative_improvement:.4f}%")





if fitness_history:
    generations = np.arange(1, len(fitness_history) + 1)

    # best_cost   = -np.asarray(fitness_history, dtype=float)
    mean_cost   = -np.asarray(mean_fitness_history, dtype=float)
    median_cost = -np.asarray(median_fitness_history, dtype=float)

    plt.figure(figsize=(9, 5))
    # plt.plot(generations, best_cost, marker='o', linewidth=1.5, label="Best (elitist) cost")
    plt.plot(generations, mean_cost, linewidth=1.5, label="Mean population cost")
    plt.plot(generations, median_cost, linewidth=1.5, label="Median population cost")

    plt.title("GA convergence diagnostics: best vs population mean/median\n"
              "Mean/median reveal whether improvement is global (healthy) or only in a few elites (risk of premature convergence).")
    plt.xlabel("Generation")
    plt.ylabel("Discounted expected cost (lower is better)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()




















# -------------------------- Report best solution ------------------------
# best_betas, best_fitness, _ = ga.best_solution()
# exp_dis_cost_best, logs = rollout_betas(best_betas)



best_betas, best_fitness, _ = ga.best_solution()

# Report: mean objective across the same fixed initial-state panel used in fitness().
costs = []
for i in range(POPULATION_SIZE):
    c, _ = rollout_betas(best_betas, state0=STATE0_POOL[i])
    costs.append(c)
exp_dis_cost_best = float(np.mean(costs))
exp_dis_cost_std  = float(np.std(costs))

# Logs: show one representative rollout (slot 0) for interpretability prints.
_, logs = rollout_betas(best_betas, state0=STATE0_POOL[0])

print(f"Best policy cost mean over STATE0_POOL: {exp_dis_cost_best:.6f} (std={exp_dis_cost_std:.6f})")



















print("betas_desc:", logs["betas_desc"])
print("min/max beta over horizon:", min(logs["beta"]), max(logs["beta"]))
print("unique actions over horizon:", sorted(set(logs["actions"])))



# Convert β thresholds to probability triggers for reporting: θ = Φ(-β)
betas_desc = _repair_thresholds(best_betas, low=ELE_GA_LB_BETA, high=ELE_GA_UB_BETA)
theta_triggers = [float(norm.cdf(-b)) for b in betas_desc]


print("\n--- Best Policy (β thresholds) ---")
print(f"β (desc): {betas_desc.tolist()}")
print(f"θ=Φ(-β):  {theta_triggers}")
print(f"Fitness (= -exp_dis_cost): {best_fitness:.6f}")
print(f"expected discounted cost: {exp_dis_cost_best:.6f}")
print(f"Actions (first 20): {logs['actions'][:20]}")

# Save compact JSON report
report = dict(
    betas_desc=betas_desc.tolist(),
    theta_desc=theta_triggers,
    J=exp_dis_cost_best,
    actions=logs["actions"],
    pf=logs["pf"],
    beta=logs["beta"],
    reward=logs["reward"],
    gamma=float(gamma),
    horizon=int(HORIZON),
    seed=int(SEED),
    pop=int(POPULATION_SIZE),
    gens=int(NUM_GENERATIONS),
    init=dict(
        mode='dirichlet' if ELE_GA_RANDOM_STATE != 'off' else 'manual',
        d0_manual=(None if ELE_GA_RESET_PROB is None else np.asarray(ELE_GA_RESET_PROB, float).ravel().tolist()),
        init_state=None,
        dirichlet_alpha=(
            None if ELE_GA_RANDOM_STATE == 'off'
            else (ELE_GA_DIRICHLET_ALPHA if np.isscalar(ELE_GA_DIRICHLET_ALPHA)
                  else np.asarray(ELE_GA_DIRICHLET_ALPHA, float).ravel().tolist())
        ),
        dirichlet_seed=None if ELE_GA_RANDOM_STATE == 'off' else int(ELE_GA_RANDOM_STATE)
    )
)

with open("pygad_beta_dp_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Wrote pygad_beta_dp_report.json")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Run completed in {elapsed_time:.2f} seconds")


# %%
# Compare the GA result with PPO result(Evaluatoin of GA policy in PPO env to have fair comparison)
# === Evaluate GA policy in the same PPO env ===
# 1) Get GA thresholds (betas_desc) from the saved report
try:
    with open("pygad_beta_dp_report.json") as f:
        ga_report = json.load(f)
        betas_desc = np.array(ga_report["betas_desc"], dtype=float)   # already DESC
except FileNotFoundError:
    print("No GA report found.")
    sys.exit(1)  # Exit with error code 1


pf_array = np.asarray(cs_pfs, float)  # shape (ncs,)

def action_policy_ga(obs, betas_desc, pf_array, ncs):
    """Map observation -> GA action via reliability thresholds."""

    # in the eval loop, curr_obs = td["observation"] is a torch.Tensor. 
    # np.asarray(torch_tensor) can produce an object array or silently behave oddly depending on dtype/device.
    # SO I need to convert to numpy explicitly.
    if torch.is_tensor(obs):
        obs_np = obs.detach().cpu().numpy()
    else:
        obs_np = np.asarray(obs, dtype=float)


    state_dis = np.asarray(obs_np[:ncs], float)
    pf_t = float(np.clip(pf_array @ state_dis, 1e-12, 1 - 1e-12))
    beta_t = float(norm.ppf(1.0 - pf_t))
    beta_ac1, beta_ac2, beta_ac3, beta_ac4 = betas_desc
    # Original version:
    # if beta_t <= beta_ac4: return 4
    # if beta_t <= beta_ac3: return 3
    # if beta_t <= beta_ac2: return 2
    # if beta_t <= beta_ac1: return 1
    # return 0 # do nothing
    # modified after discussion with David
    if beta_t > beta_ac1:
        action = 0  # do nothing
    elif beta_ac2 <= beta_t < beta_ac1:
        action = 1  # maintenance
    elif beta_ac3 <= beta_t < beta_ac2:
        action = 2  # repair
    elif beta_ac4 <= beta_t < beta_ac3:
        action = 3  # rehabilitation
    elif beta_t <= beta_ac4:
        action = 4  # replacement
    else:
        raise ValueError(f"Unexpected beta_t={beta_t} with thresholds={betas_desc}")
    return action    


# load constants
horizon = HORIZON
n_episodes = test_constants.ELE_GA_N_EPISODES_EVAL
max_cost = test_constants.ELE_GA_MAX_COST_EVAL
reset_prob = test_constants.ELE_GA_RESET_PROB_EVAL
dirichlet_alpha = test_constants.ELE_GA_DIRICHLET_ALPHA_EVAL
random_state = test_constants.ELE_GA_RANDOM_STATE_EVAL
explore_type = test_constants.ELE_GA_EXPLORE_TYPE_EVAL
include_step = test_constants.ELE_GA_INC_STEP_EVAL

print("horizon:", horizon)
print("n_episodes:", n_episodes)
print("max_cost:", max_cost)
print("reset_prob:", reset_prob)
print("dirichlet_alpha:", dirichlet_alpha)
print("random_state:", random_state)
print("explore_type:", explore_type)
print("include_step:", include_step)

# 2) Recreate the same env config we used for PPO/DP eval to have fair comparison
env = create_element_env(
    horizon,
    max_cost=max_cost,
    include_step_count=include_step,
    reset_prob=reset_prob,
    dirichlet_alpha=dirichlet_alpha,
    random_state=random_state
)

# 3) Roll out multiple episodes and average reward (same as DP loop and PPO eval)
logs = defaultdict(list)
with tqdm(total=n_episodes*horizon) as pbar:
    with set_exploration_type(explore_type), torch.no_grad():
        for _ in range(n_episodes):
            td = env.reset()

            obs_len = int(td["observation"].numel())
            observation = np.zeros((horizon, obs_len), dtype=np.float32)
            action      = np.zeros((horizon, 1),   dtype=np.int64)
            reward      = np.zeros((horizon, 1),   dtype=np.float32)

            init_obs = td["observation"]
            init_time_idx = int(init_obs[-1].item() * horizon) if include_step else 0

            for i in range(init_time_idx, horizon):
                curr_obs = td["observation"]
                a = action_policy_ga(curr_obs, betas_desc, pf_array, ncs)
                td["action"] = torch.tensor(a, dtype=torch.int64)


                res = env.step(td)
                observation[i]   = res["observation"].cpu().numpy()
                action[i]        = res["action"].cpu().numpy()
                reward[i]        = res["next","reward"].cpu().numpy()
                td["observation"] = res["next","observation"]


            logs["observation"].append(observation)
            logs["action"].append(action)
            logs["reward"].append(reward)
            logs["ep reward"].append(reward.sum().item())
            pbar.update(horizon)


# print(f"Average ep reward (GA policy): {np.mean(logs['ep reward']):.4f}")
ep_rewards = np.asarray(logs["ep reward"], dtype=float)
mean_ep_reward = ep_rewards.mean()
std_ep_reward  = ep_rewards.std()

print(
    f"Average ep reward (GA policy): "
    f"{mean_ep_reward:.4f} ± {std_ep_reward:.4f}"
)


#%%
# action distribution summary
all_actions = np.concatenate(logs["action"])
id2name = {0:"Do nothing",1:"Maintenance",2:"Repair",3:"Rehabilitation",4:"Replacement"}
unique, counts = np.unique(all_actions, return_counts=True)
action_distribution_ga = {id2name.get(int(a), int(a)): int(c) for a, c in zip(unique, counts)}
print("GA action distribution:", action_distribution_ga)

# Action sequence for the single evaluation episode (t = 0..T-1 for steps actually taken) 
ep0_actions = logs["action"][0].astype(int).flatten()          # shape: (horizon,)
ep0_obs     = logs["observation"][0]                           # shape: (horizon, obs_len)

ep0_action_names = [id2name.get(int(a), str(int(a))) for a in ep0_actions]
print("\nEvaluation action sequence (time-ordered):")
print(", ".join(ep0_action_names))

# Condition-state distribution per step (episode 0)
ep0_obs = logs["observation"][0]                 # shape: (horizon, obs_dim)
obs_dim = ep0_obs.shape[1]

# If include_step_count=True, the last obs entry is normalized time; otherwise there is no time column.
ncs_eff = int(obs_dim - (1 if include_step else 0))   # number of condition-state entries

cs_traj = ep0_obs[:, :ncs_eff]                                # (horizon, ncs)
t_traj  = ep0_obs[:, ncs_eff] if include_step else None





print("\nCondition-state distribution per step (episode 0):")
for t, (cs, a) in enumerate(zip(cs_traj, ep0_actions)):
    cs_str = ", ".join([f"cs{k}={p:.3f}" for k, p in enumerate(cs)])
    if t_traj is not None:
        # τ is the normalized time in [0,1]
        print(f"Step={t:02d}  t={t_traj[t]:.3f}  action={id2name[int(a)]:<14} [{cs_str}]")
    else:
        print(f"Step={t:02d}  act={id2name[int(a)]:<14} [{cs_str}]")









print("\nCondition-state distribution per step (first 3 episodes):")

n_print = min(5, len(logs["observation"]))

for ep in range(n_print):
    print(f"\n=== Episode {ep} ===")

    ep_obs     = logs["observation"][ep]                  # (horizon, obs_dim)
    ep_actions = logs["action"][ep].astype(int).flatten()  # (horizon,)

    # --- rewards (per-step), return, avg ---
    ep_rewards = logs["reward"][ep].astype(float).flatten()
    ep_return  = ep_rewards.sum()
    ep_avg     = ep_rewards.mean()

    # --- initial state (from step 0 observation) ---
    init_state = ep_obs[0, :ncs]   # first ncs entries are CS distribution
    init_str = ", ".join([f"cs{k}={p:.3f}" for k, p in enumerate(init_state)])

    print(f"Episode {ep}: return(sum)={ep_return:.4f} | avg/step={ep_avg:.6f}")
    print(f"Initial state: [{init_str}]")

    # --- per-step CS distribution + action ---
    obs_dim = ep_obs.shape[1]
    ncs_eff = int(obs_dim - (1 if include_step else 0))

    cs_traj = ep_obs[:, :ncs_eff]
    t_traj  = ep_obs[:, ncs_eff] if include_step else None

    for t, (cs, a) in enumerate(zip(cs_traj, ep_actions)):
        cs_str = ", ".join([f"cs{k}={p:.3f}" for k, p in enumerate(cs)])
        if t_traj is not None:
            print(f"Step={t:02d}  t={t_traj[t]:.3f}  action={id2name[int(a)]:<14} [{cs_str}]")
        else:
            print(f"Step={t:02d}  action={id2name[int(a)]:<14} [{cs_str}]")











# %%
# I Generated the following part by AI
# 1) action sequence for the single evaluation episode
ep0_actions = logs["action"][0].astype(int).flatten()
T = len(ep0_actions)

# 2) color palette (crisp, colorblind-friendly-ish)
action_colors = {
    0: "#9E9E9E",  # gray
    1: "#4E79A7",  # blue
    2: "#59A14F",  # green
    3: "#F28E2B",  # orange
    4: "#E15759",  # red
}
colors = [action_colors[int(a)] for a in ep0_actions]

# 3) plot a single horizontal bar split into T segments
fig, ax = plt.subplots(figsize=(max(10, T*0.35), 1.6))
lefts = np.arange(T)
ax.barh(
    y=0, width=np.ones(T), left=lefts, height=0.85,
    color=colors, edgecolor="white", linewidth=1.4
)

# 4) label each segment with the step number (1..T) centered, with auto-contrasting text
for t, c in enumerate(colors):
    r, g, b = mcolors.to_rgb(c)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    txt_color = "white" if luminance < 0.5 else "black"
    ax.text(t + 0.5, 0, str(t+1), ha="center", va="center", fontsize=9, color=txt_color, fontweight="bold")

# 5) cosmetics: GA label on the left, no title, clean axes
ax.set_ylim(-0.8, 0.8)
ax.set_xlim(0, T)
ax.set_yticks([0])
ax.set_yticklabels(["GA"], fontsize=11)   # put "GA" on the left
ax.set_xticks([])                         # numbers are inside each segment already
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

# optional legend (comment out if you don't want it)
handles = [Patch(facecolor=action_colors[k], label=id2name[k]) for k in sorted(action_colors)]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", context="talk")

beta_min = 2.5
beta_max = 4.2

gap = (beta_max - beta_min) / 5.0
beta_levels = beta_max - gap * np.arange(1, 5)
labels = ["Beta1", "Beta2", "Beta3", "Beta4"]

steps = np.linspace(0, 10, 200)
beta_bridge = beta_max - (beta_max - beta_min) * (steps / steps.max())**2

plt.figure(figsize=(10, 6))

for beta, label in zip(beta_levels, labels):
    plt.hlines(beta, xmin=0, xmax=10,
               linestyles='--', linewidth=2, color='black')
    plt.text(10.2, beta, label, va='center', color='black')

plt.plot(steps, beta_bridge, linewidth=3, color='black')

plt.xlabel("step")
plt.ylabel("β_bridge")

plt.ylim(beta_min, beta_max)
plt.xlim(0, 10)

plt.grid(False)

sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()

# %%
