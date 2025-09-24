# utility functions
import networkx as nx


def cost_util(cost, min_cost=0, max_cost=1):
    # util = (max_cost-cost)/(max_cost-min_cost)
    # if util < 0:
    #     util = 0
    # elif util > 1:
    #     util = 1
    util = -cost/max_cost
    return util
