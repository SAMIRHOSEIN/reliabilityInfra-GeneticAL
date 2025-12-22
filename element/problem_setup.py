# #%%
# # #################################################
# # 1. Original enviroment setup with 5 condition states
# # #################################################
# # Original model with 5 condition states
# # Simple network with 4 nodes, 5 links, and 3 bridges
# import numpy as np
# import scipy.stats as stats
# import networkx as nx

# ncs, na, gamma, horizon = 5, 5, 1/1.03, 200

# p11, p22, p33, p44 = 0.9381, 0.8888, 0.8712, 0.8888
# p12, p23, p34, p45 = 0.0619, 0.1112, 0.1288, 0.1112

# cost_base = 10

# cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5, -2.0])
# failure_cost = cost_base**7

# # action_model(s, a, s') = the probability of transitioning from s to s' given a
# # do nothing
# action0 = np.array([
#     [p11, p12, 0, 0, 0],
#     [0, p22, p23, 0, 0],
#     [0, 0, p33, p34, 0],
#     [0, 0, 0, p44, p45],
#     [0, 0, 0, 0, 1]
# ])
# unit_price0 = np.zeros(ncs)

# # maintenance (with little to no condition improvement)
# action1 = np.array([
#     [0.99, 0.01, 0, 0, 0],
#     [0.03, 0.95, 0.02, 0, 0],
#     [0, 0.03, 0.95, 0.02, 0],
#     [0, 0, 0.03, 0.95, 0.02],
#     [0, 0, 0, 0, 1]
# ])
# unit_price1 = np.array([cost_base**2, cost_base**2, cost_base**2, cost_base**2, 0])

# # repair (mostly raise condition by 1)
# action2 = np.array([
#     [0.99, 0.01, 0, 0, 0],
#     [0.5, 0.45, 0.05, 0, 0],
#     [0, 0.5, 0.45, 0.05, 0],
#     [0, 0, 0.5, 0.45, 0.05],
#     [0, 0, 0, 0.5, 0.5],
# ])
# unit_price2 = np.array([cost_base**3, cost_base**3, cost_base**4, cost_base**4, cost_base**5])

# # rehabilitation (mostly raise condition by 2)
# action3 = np.array([
#     [0.99, 0.01, 0, 0, 0],
#     [0.5, 0.45, 0.05, 0, 0],
#     [0.5, 0.3, 0.2, 0, 0],
#     [0.4, 0.3, 0.2, 0.1, 0],
#     [0.4, 0.3, 0.2, 0.1, 0],
# ])
# unit_price3 = np.array([cost_base**4, cost_base**4, cost_base**5, cost_base**5, cost_base**6])

# # replacement
# action4 = np.array([
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
# ])
# unit_price4 = np.array([cost_base**7, cost_base**7, cost_base**7, cost_base**7, cost_base**7])

# action_model = np.array([action0, action1, action2, action3, action4])
# unit_costs = np.array([unit_price0, unit_price1, unit_price2, unit_price3, unit_price4])


# # #################################################
# # 2. My first protocol: merging CS4 and CS5
# # #################################################

# # Modified model with 4 condition states
# import numpy as np
# import scipy.stats as stats

# # NEW MODEL: 4 CONDITION STATES
# ncs, na, gamma, horizon = 4, 5, 1/1.03, 200

# # Transition parameters(same as original model)
# p11, p22, p33, p44 = 0.9381, 0.8888, 0.8712, 0.8888
# p12, p23, p34, p45 = 0.0619, 0.1112, 0.1288, 0.1112

# cost_base = 10

# # Failure probabilities
# # ============================
# # Failure probabilities
# # Build 5-CS probs, then compress to 4-CS
# # ============================
# # we conservatively adopt the maximum failure rate among the merged severe states CS4–CS5.
# cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  
# # # cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.0])  
# # cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.25])

# failure_cost = cost_base**7

# # ACTION MODELS REDUCED FROM 5x5 -> 4x4
# def drop_cs5(M5):
#     """
#     Convert 5x5 transition matrix to 4x4 by:
#     - Keeping rows 0..3 (drop row 4)
#     - Merging column 4 into column 3
#     - Dropping column 4
#     """
#     # Merge column 4 into column 3
#     M4 = M5[:, :4].copy()
#     M4[:, 3] += M5[:, 4]

#     # Now drop row 4
#     M4 = M4[:4, :]

#     return M4

# # ACTION 0 — Do nothing
# action0_5 = np.array([
#     [p11, p12, 0,    0,    0],
#     [0,   p22, p23, 0,    0],
#     [0,   0,   p33, p34,  0],
#     [0,   0,   0,   p44,  p45],
#     [0,   0,   0,   0,    1]
# ])
# action0 = drop_cs5(action0_5)
# unit_price0 = np.zeros(4)   # dropped the CS5 cost (was zero anyway)

# # ACTION 1 — Maintenance
# action1_5 = np.array([
#     [0.99, 0.01, 0,    0,    0],
#     [0.03, 0.95, 0.02, 0,    0],
#     [0,    0.03, 0.95, 0.02, 0],
#     [0,    0,    0.03, 0.95, 0.02],
#     [0,    0,    0,    0,    1]
# ])
# action1 = drop_cs5(action1_5)
# unit_price1 = np.array([cost_base**2]*4)   # drop CS5 cost

# # ACTION 2 — Repair
# action2_5 = np.array([
#     [0.99, 0.01, 0,    0,    0],
#     [0.5,  0.45, 0.05, 0,    0],
#     [0,    0.5,  0.45, 0.05, 0],
#     [0,    0,    0.5,  0.45, 0.05],
#     [0,    0,    0,    0.5,  0.5],
# ])
# action2 = drop_cs5(action2_5)
# unit_price2 = np.array([cost_base**3, cost_base**3, cost_base**4, cost_base**4])  # drop CS5

# # ACTION 3 — Rehabilitation
# action3_5 = np.array([
#     [0.99, 0.01, 0,    0,    0],
#     [0.5,  0.45, 0.05, 0,    0],
#     [0.5,  0.3,  0.2,  0,    0],
#     [0.4,  0.3,  0.2,  0.1,  0],
#     [0.4,  0.3,  0.2,  0.1,  0],
# ])
# action3 = drop_cs5(action3_5)
# unit_price3 = np.array([cost_base**4, cost_base**4, cost_base**5, cost_base**5])  # drop CS5

# # ACTION 4 — Replacement
# action4_5 = np.array([
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
#     [1.0, 0, 0, 0, 0],
# ])
# action4 = drop_cs5(action4_5)
# unit_price4 = np.array([cost_base**7]*4)  # drop CS5

# # PACK INTO FINAL ARRAYS
# action_model = np.array([action0, action1, action2, action3, action4])
# unit_costs = np.array([unit_price0, unit_price1, unit_price2, unit_price3, unit_price4])

# # %%


# ##################################################################################################
# 3. AASHTO 2010_V1 (https://www.google.com/url?sa=D&q=https://apmgs.ro/files/documente/AASHTO-bridge_element_guide_manual__05092010.pdf&ust=1766092560000000&usg=AOvVaw3ttCFCe2ufe4c431wOqeMq&hl=en)
# merging CS2 and CS3 -> ACS2: Page: 167 in section D.2.1
# ##################################################################################################

# Modified model with 4 condition states
import numpy as np
import scipy.stats as stats

# NEW MODEL: 4 CONDITION STATES
ncs, na, gamma, horizon = 4, 5, 1/1.03, 200

# Transition parameters(same as original model)
p11, p22, p33, p44 = 0.9381, 0.8888, 0.8712, 0.8888
p12, p23, p34, p45 = 0.0619, 0.1112, 0.1288, 0.1112

cost_base = 10

# Failure probabilities
# ============================
# Failure probabilities
# Build 5-CS probs, then compress to 4-CS
# ============================
# we conservatively adopt the maximum failure rate among the merged severe states CS4–CS5.
cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  
# # cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.0])  
# cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.25])
# cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -1])  

failure_cost = cost_base**7

# ACTION 0 — Do nothing
action0 = np.array([
    [0.9381, 0.0619, 0, 0],
    [0, 0.9356, 0.0644, 0],
    [0, 0, 0.8888, 0.1112],
    [0, 0, 0, 1]
])
unit_price0 = np.zeros(4)   # dropped the CS5 cost (was zero anyway)

# ACTION 1 — Maintenance
action1 = np.array([
    [0.99, 0.01, 0, 0],
    [0.015, 0.975, 0.01, 0],
    [0, 0.03, 0.95, 0.02],
    [0, 0, 0, 1]
])
unit_price1 = np.array([cost_base**2]*4)   # drop CS5 cost

# ACTION 2 — Repair
action2 = np.array([
    [0.99, 0.01, 0, 0],
    [0.25, 0.725, 0.025, 0],
    [0, 0.5, 0.45, 0.05],
    [0, 0, 0.5, 0.5]
])
unit_price2 = np.array([cost_base**3, cost_base**3, cost_base**4, cost_base**4])  # drop CS5

# ACTION 3 — Rehabilitation
action3 = np.array([
    [0.99, 0.01, 0, 0],
    [0.5, 0.5, 0, 0],
    [0.4, 0.5, 0.1, 0],
    [0.4, 0.5, 0.1, 0]
])
unit_price3 = np.array([cost_base**4, cost_base**4, cost_base**5, cost_base**5])  # drop CS5

# ACTION 4 — Replacement
action4 = np.array([
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0]
])
unit_price4 = np.array([cost_base**7]*4)  # drop CS5

# PACK INTO FINAL ARRAYS
action_model = np.array([action0, action1, action2, action3, action4])
unit_costs = np.array([unit_price0, unit_price1, unit_price2, unit_price3, unit_price4])

# %%