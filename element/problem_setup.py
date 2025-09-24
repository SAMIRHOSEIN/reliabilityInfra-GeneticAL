# %%
# Simple network with 4 nodes, 5 links, and 3 bridges
import numpy as np
import scipy.stats as stats
import networkx as nx

ncs, na, gamma, horizon = 5, 5, 1/1.03, 200

p11, p22, p33, p44 = 0.9381, 0.8888, 0.8712, 0.8888
p12, p23, p34, p45 = 0.0619, 0.1112, 0.1288, 0.1112

cost_base = 10

cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5, -2.0])
failure_cost = cost_base**7

# action_model(s, a, s') = the probability of transitioning from s to s' given a
# do nothing
action0 = np.array([
    [p11, p12, 0, 0, 0],
    [0, p22, p23, 0, 0],
    [0, 0, p33, p34, 0],
    [0, 0, 0, p44, p45],
    [0, 0, 0, 0, 1]
])
unit_price0 = np.zeros(ncs)

# maintenance (with little to no condition improvement)
action1 = np.array([
    [0.99, 0.01, 0, 0, 0],
    [0.03, 0.95, 0.02, 0, 0],
    [0, 0.03, 0.95, 0.02, 0],
    [0, 0, 0.03, 0.95, 0.02],
    [0, 0, 0, 0, 1]
])
unit_price1 = np.array([cost_base**2, cost_base**2, cost_base**2, cost_base**2, 0])

# repair (mostly raise condition by 1)
action2 = np.array([
    [0.99, 0.01, 0, 0, 0],
    [0.5, 0.45, 0.05, 0, 0],
    [0, 0.5, 0.45, 0.05, 0],
    [0, 0, 0.5, 0.45, 0.05],
    [0, 0, 0, 0.5, 0.5],
])
unit_price2 = np.array([cost_base**3, cost_base**3, cost_base**4, cost_base**4, cost_base**5])

# rehabilitation (mostly raise condition by 2)
action3 = np.array([
    [0.99, 0.01, 0, 0, 0],
    [0.5, 0.45, 0.05, 0, 0],
    [0.5, 0.3, 0.2, 0, 0],
    [0.4, 0.3, 0.2, 0.1, 0],
    [0.4, 0.3, 0.2, 0.1, 0],
])
unit_price3 = np.array([cost_base**4, cost_base**4, cost_base**5, cost_base**5, cost_base**6])

# replacement
action4 = np.array([
    [1.0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0],
])
unit_price4 = np.array([cost_base**7, cost_base**7, cost_base**7, cost_base**7, cost_base**7])

action_model = np.array([action0, action1, action2, action3, action4])
unit_costs = np.array([unit_price0, unit_price1, unit_price2, unit_price3, unit_price4])