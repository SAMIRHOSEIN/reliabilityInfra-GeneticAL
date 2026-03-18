"""
Parameters of NBE 107 deterioation and maintanance

The variables are based on the following assumptions:
1. Deterioration transition matrix is from Thompson et al. (1998) Table 2 (CoRe element 107)
2. ASHTO Bridge Element Inspection Guide Manual (2010) provides migration from CoRe to MBE.
   Based on Section D.2.1 (page 167), we merge CS2 and CS3 of CoRe to CS2 of NBE.
3. When merging Markov states, assume CoRe CS2 and CS3 have equal weights
   (i.e., equal likelihood of occurring).
4. The transition probabilities in CoRe 107 are combined with the following equation:
   p(nbe2->nbe3) = q(core2->core4) * 0.5 + q(core3->core4) * 0.5 = q(core3->core4) * 0.5
5. Failure probablities given all elements are in a CS are assumed from common target reliability indexes.
6. Cost data are assumed: unit_prices are per element cost. They can be tied to CS before action.
"""
import numpy as np
import scipy.stats as stats


__all__ = [
    "NCS", "NA",
    "CS_PFS", "FAILURE_COST",
    "ACTION_MODEL", "UNIT_COSTS",
]


# RL parameters
NCS, NA = 4, 5

# Failure probabilities
CS_PFS = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  
cost_base = 10
FAILURE_COST = cost_base**7

# Action 0 — Do nothing
action0 = np.array([
    [0.9381, 0.0619, 0, 0],
    [0, 0.9356, 0.0644, 0],
    [0, 0, 0.8888, 0.1112],
    [0, 0, 0, 1]
])
unit_price0 = np.zeros(NCS)

# Action 1 — Maintenance
action1 = np.array([
    [0.99, 0.01, 0, 0],
    [0.015, 0.975, 0.01, 0],
    [0, 0.03, 0.95, 0.02],
    [0, 0, 0, 1]
])
unit_price1 = np.array([cost_base**2]*4)

# Action 2 — Repair
action2 = np.array([
    [0.99, 0.01, 0, 0],
    [0.25, 0.725, 0.025, 0],
    [0, 0.5, 0.45, 0.05],
    [0, 0, 0.5, 0.5]
])
unit_price2 = np.array([cost_base**3, cost_base**3,
                        cost_base**4, cost_base**4])

# Action 3 — Rehabilitation
action3 = np.array([
    [0.99, 0.01, 0, 0],
    [0.5, 0.5, 0, 0],
    [0.4, 0.5, 0.1, 0],
    [0.4, 0.5, 0.1, 0]
])
unit_price3 = np.array([cost_base**4, cost_base**4,
                        cost_base**5, cost_base**5])

# Action 4 — Replacement
action4 = np.array([
    [1.0, 0, 0, 0],
    [1.0, 0, 0, 0],
    [1.0, 0, 0, 0],
    [1.0, 0, 0, 0]
])
unit_price4 = np.array([cost_base**7]*NCS)  # drop CS5

# Pack into final arrays
ACTION_MODEL = np.array([action0, action1, action2, action3, action4])
UNIT_COSTS = np.array([unit_price0, unit_price1, unit_price2, unit_price3, unit_price4])
