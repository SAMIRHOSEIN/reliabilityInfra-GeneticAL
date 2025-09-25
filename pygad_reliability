# %%
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Final version of the PyGAD script
#-------------------------------------------------------------------------------------------------------------------------------------------------
import json 
import numpy as np
from scipy.stats import norm
import pygad
import time
start_time = time.time()
# -------------------------- Load problem setup --------------------------
from element.problem_setup import (
    ncs, na, gamma,
    action_model, unit_costs, cs_pfs, failure_cost
)

from element.utility_func import cost_util
# -------------------------- GA & model knobs (ALL HERE) -----------------
import test_constants

#load constants
SEED = test_constants.ELE_GA_SEED_FOR_PyGAD
NUM_GENERATIONS = test_constants.ELE_GA_GENS
POPULATION_SIZE = test_constants.ELE_GA_POP
NUM_PARENTS_MATING = max(2, POPULATION_SIZE // 2) # ensure at least 2 parents
MUTATION_PERCENT_GENES = test_constants.ELE_GA_MUTATION_PERCENT_GENES
CROSSOVER_TYPE = test_constants.ELE_GA_CROSSOVER_TYPE
PARENT_SELECTION = test_constants.ELE_GA_PARENT_SELECTION
KEEP_PARENTS = test_constants.ELE_GA_KEEP_PARENTS
ELE_GA_LB_BETA = test_constants.ELE_GA_LB_BETA
ELE_GA_UB_BETA = test_constants.ELE_GA_UB_BETA
ELE_GA_MAX_COST = test_constants.ELE_GA_MAX_COST
ELE_GA_RANDOM_STATE = test_constants.ELE_GA_RANDOM_STATE
ELE_GA_RESET_PROB = test_constants.ELE_GA_RESET_PROB
ELE_GA_DIRICHLET_ALPHA = test_constants.ELE_GA_DIRICHLET_ALPHA
horizon = test_constants.ELE_GA_HORIZON
# -------------------------- Prepare arrays ------------------------------
# Keep the same shapes used in rl_env.step():
P_actions = np.asarray(action_model, float)   # (na, ncs, ncs)
Unit_costs = np.asarray(unit_costs, float)    # (na, ncs)  (we'll index Unit_costs[action] @ state)
pf_array   = np.clip(np.asarray(cs_pfs, float), 1e-12, 1 - 1e-12)  # (ncs,)

# -------------------------- Build STATE0 once (reset) -------------------
if ELE_GA_RANDOM_STATE == 'off':
    assert ELE_GA_RESET_PROB is not None, "ELE_GA_RESET_PROB must be set when RANDOM_STATE is 'off'."
    STATE0 = np.asarray(ELE_GA_RESET_PROB, float).ravel()
    assert STATE0.shape[0] == ncs and np.isclose(STATE0.sum(), 1.0), "ELE_GA_RESET_PROB must sum to 1."
else:
    assert ELE_GA_DIRICHLET_ALPHA is not None, "ELE_GA_DIRICHLET_ALPHA must be set when RANDOM_STATE is used."
    rng = np.random.default_rng(ELE_GA_RANDOM_STATE)
    STATE0 = rng.dirichlet(np.asarray(ELE_GA_DIRICHLET_ALPHA, float).ravel())

# -------------------------- Policy & rollout ----------------------------
def reliability_based_action(beta_t, betas_desc):
    """betas_desc is sorted DESC: [β1 ≥ β2 ≥ β3 ≥ β4]."""
    beta_ac1, beta_ac2, beta_ac3, beta_ac4 = betas_desc
    if beta_t <= beta_ac4: return 4
    if beta_t <= beta_ac3: return 3
    if beta_t <= beta_ac2: return 2
    if beta_t <= beta_ac1: return 1
    return 0 # do nothing


def rollout_betas(betas_raw):
    """
    Simulate one episode over the finite horizon using β-thresholds.
    Returns:
      exp_dis_cost (float): expected discounted cost we minimize
      logs (dict): actions, pf, beta, reward series for inspection
    """
    betas_raw = np.asarray(betas_raw, float)
    betas_raw = np.clip(betas_raw, ELE_GA_LB_BETA, ELE_GA_UB_BETA) #clip due to numerical stability and if beta was very large/small, we will face issue due to np.cdf
    betas_desc = np.sort(betas_raw)[::-1]
    
    state = STATE0.copy()   # shape (ncs,), initial state distribution = STATE0 

    H = int(horizon)
    exp_dis_cost = 0.0
    actions, pf_series, beta_series, reward_series = [], [], [], []

    for t in range(H):
      # prob of failure and reliability index at time t
      pf_t= float(np.clip(pf_array @ state, 1e-12, 1-1e-12))    #clip due to numerical stability(norm.ppf(1) and norm.ppf(0) are inf) and keep the serch practical.
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
      state = state / state.sum()  # normalize to fix any rounding

    logs = dict(
        betas_desc=betas_desc.tolist(),
        actions=actions,
        pf=pf_series,
        beta=beta_series,
        reward=reward_series,
        exp_dis_cost=float(exp_dis_cost)
    )

    return exp_dis_cost, logs
      
# -------------------------- PyGAD fitness-------------
#must include all three parameters because PyGAD will pass them,
def fitness_func(ga_instance, solution, solution_idx):
    exp_dis_cost, _ = rollout_betas(solution)
    return float(-exp_dis_cost)  # PyGAD maximizes fitness; we minimize cost.

# -------------------------- Run GA --------------------------------------
na_exc_zero = na - 1  # num of actions excluding 'do nothing' (action 0)
gene_space = [{'low': ELE_GA_LB_BETA, 'high': ELE_GA_UB_BETA} for _ in range(na_exc_zero)]

ga = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    sol_per_pop=POPULATION_SIZE,
    num_genes=na_exc_zero,
    gene_space=gene_space,
    init_range_low=ELE_GA_LB_BETA,
    init_range_high=ELE_GA_UB_BETA,
    mutation_type="random",
    mutation_percent_genes=MUTATION_PERCENT_GENES,
    parent_selection_type=PARENT_SELECTION,
    crossover_type=CROSSOVER_TYPE,
    keep_parents=KEEP_PARENTS,
    random_seed=SEED
)

print("PyGAD β-threshold optimization (DP objective)")
ga.run()

# -------------------------- Report best solution ------------------------
best_betas, best_fitness, _ = ga.best_solution()
exp_dis_cost_best, logs = rollout_betas(best_betas)

# Convert β thresholds to probability triggers for reporting: θ = Φ(-β)
betas_desc = np.sort(np.clip(best_betas, ELE_GA_LB_BETA, ELE_GA_UB_BETA))[::-1]
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
    horizon=int(horizon),
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
import numpy as np
import json
import torch
from scipy.stats import norm
from collections import defaultdict
from tqdm import tqdm
from torchrl.envs.utils import set_exploration_type
from torchrl_bridge import create_element_env
from element.problem_setup import ncs, cs_pfs
import test_constants
import sys

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
    state_dis = np.asarray(obs[:ncs], float)
    pf_t = float(np.clip(pf_array @ state_dis, 1e-12, 1 - 1e-12))
    beta_t = float(norm.ppf(1.0 - pf_t))
    beta_ac1, beta_ac2, beta_ac3, beta_ac4 = betas_desc
    if beta_t <= beta_ac4: return 4
    if beta_t <= beta_ac3: return 3
    if beta_t <= beta_ac2: return 2
    if beta_t <= beta_ac1: return 1
    return 0 # do nothing


# load constants
horizon = test_constants.ELE_GA_HORIZON
n_episodes = test_constants.ELE_GA_N_EPISODES_EVAL
max_cost = test_constants.ELE_GA_MAX_COST_EVAL
reset_prob = test_constants.ELE_GA_RESET_PROB_EVAL
dirichlet_alpha = test_constants.ELE_GA_DIRICHLET_ALPHA_EVAL
random_state = test_constants.ELE_GA_RANDOM_STATE_EVAL
explore_type = test_constants.ELE_GA_EXPLORE_TYPE_EVAL
include_step = test_constants.ELE_GA_INC_STEP_EVAL

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


print(f"Average ep reward (GA policy): {np.mean(logs['ep reward']):.4f}")

# action distribution summary
all_actions = np.concatenate(logs["action"])
id2name = {0:"Do nothing",1:"Maintenance",2:"Repair",3:"Rehabilitation",4:"Replacement"}
unique, counts = np.unique(all_actions, return_counts=True)
action_distribution_ga = {id2name.get(int(a), int(a)): int(c) for a, c in zip(unique, counts)}
print("GA action distribution:", action_distribution_ga)
# %%
