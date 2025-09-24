# %%
import numpy as np
import torch
from torch import nn

# environment
from torchrl.envs import GymWrapper
from element.problem_setup import ncs, na
from element.problem_setup import gamma as discount
from element.problem_setup import action_model, unit_costs, failure_cost
from element.rl_env import SingleElement


# constants for creete_element_env
_RESET_PROB = np.array([1.0]+[0.0]*(ncs-1))
_MAX_COST = 1.0

# constants for ConstantActionModule
_CONST_ACTION_DEFAULT = 0


def create_element_env(
    horizon,
    include_step_count=False,
    max_cost=_MAX_COST,
    reset_prob=_RESET_PROB,
    dirichlet_alpha=None,
    random_state='off'
):
    max_step = horizon

    base_env = SingleElement(
        state_size=ncs, action_size=na,
        max_step=max_step,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        dirichlet_alpha=dirichlet_alpha,
        action_model=action_model,
        unit_costs=unit_costs, 
        cost_params=(None, max_cost),
        discount=discount,
        failure_cost=failure_cost,
        random_state=random_state,
    )

    env = GymWrapper(base_env, categorical_action_encoding=True)

    return env


class ValueNet(nn.Module):
    def __init__(
        self, input_dim,
        value_cells, value_layers,
        device=torch.device("cpu")
    ):
        # no need for input_dim due to LazyLinear
        super().__init__()
        layers = [nn.Linear(input_dim, value_cells, device=device), nn.ELU()]
        layers = layers + [nn.Linear(value_cells, value_cells, device=device), nn.ELU()] * value_layers
        layers.append(nn.Linear(value_cells, 1, device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConstantModule(nn.Module):
    def __init__(self, constant_value: int = _CONST_ACTION_DEFAULT):
        super().__init__()
        self.constant = torch.as_tensor(constant_value)

    def forward(self, x):
        # ignore input x and always return the constant
        return self.constant


class ElementActorNet(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        actor_cells, actor_layers,
        device=torch.device("cpu")
    ):
        # Change LazyLinear to Linear
        super().__init__()
        layers = [nn.Linear(input_dim, actor_cells, device=device), nn.ELU()]
        layers = layers + [nn.Linear(actor_cells, actor_cells, device=device), nn.ELU()] * actor_layers
        layers.append(nn.Linear(actor_cells, output_dim, device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#TODO: implement softtree as an actor module
class ElementActorTree(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
# %%