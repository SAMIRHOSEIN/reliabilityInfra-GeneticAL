# %%
import numpy as np

# # openai/gym imports
# import gym
# from gym import spaces

# gynasium imports
import gymnasium as gym
from gymnasium import spaces

from element.problem_setup import ncs, na, cs_pfs, \
    action_model, unit_costs, failure_cost
from element.utility_func import cost_util


_MAX_STEP = 100
_DISCOUNT = 1.0

class SingleElement(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    pf_array = cs_pfs

    def __init__(
        self, state_size=ncs, action_size=na,
        include_step_count=False,
        max_step=_MAX_STEP,
        reset_prob=None,
        dirichlet_alpha=None,
        action_model=action_model,
        unit_costs=unit_costs, cost_params=(None, None),
        failure_cost=failure_cost,
        discount=_DISCOUNT,
        render_mode=None,
        random_state: int | str | None = None,
    ):
        # store env parameters
        self.state_size = state_size
        self.action_size = action_size
        self.max_step = max_step
        self.include_step_count = include_step_count

        self.discount = discount
        self.failure_cost = failure_cost
        self.action_model, self.unit_costs = action_model, unit_costs
        if cost_params[0] is None:
            self.min_cost = 0
        else:
            self.min_cost = cost_params[0]
        assert cost_params[1] is not None, "Must provide maximum cost in cost_params"
        self.max_cost = cost_params[1]

        if reset_prob is None:
            assert dirichlet_alpha is not None, "Must provide dirichlet_alpha if no reset_prob"
            assert random_state != 'off', "Must provide random_state if no reset_prob"
            self.reset_prob = None
            self.dirichlet_alpha = dirichlet_alpha
        else:
            assert dirichlet_alpha is None, "Cannot provide dirichlet_alpha if reset_prob is provided"
            self.reset_prob = reset_prob
            self.dirichlet_alpha = None

        # define observation and action spaces
        obs_size = self.state_size+1 if include_step_count else self.state_size
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.random_state = random_state
        if random_state != 'off':
            super().reset(seed=self.random_state)

        # initialize hidden state and parameters
        self.reset()
    
    def reset(self, options=None):
        self._time = 0
        if self.random_state == 'off':
            self._state = self.reset_prob
        else:
            state = self.np_random.dirichlet(
                self.dirichlet_alpha,
            )
            self._state = state.astype(np.float32)


        if self.include_step_count:
            observation = np.append(self._state, self._time/self.max_step)
        else:
            observation = self._state
        info = {"cs": self._state, "time": self._time}

        return observation, info

    def step(self, action):
        # state after the action
        state = self.action_model[int(action)].T @ self._state
        # normalize state to fix numerical errors
        state = state / state.sum()

        # reward due to action
        dir_cost = self.unit_costs[int(action)] @ self._state
        fail_risk = (self.pf_array @ self._state) * self.failure_cost
        cost = dir_cost + fail_risk
        reward = cost_util(cost, min_cost=self.min_cost, max_cost=self.max_cost)
        discount_factor = self.discount**self._time
        reward = np.float32(discount_factor * reward)


        # update hidden state
        self._state = state
        self._time += 1

        # update observation
        if self.include_step_count:
            observation = np.append(state, self._time/self.max_step)
        else:
            observation = state

        # check if done
        if self._time >= self.max_step:
            done = True
            _, info = self.reset()
        else:
            done = False
            info = {
                "cs": state, "dir_cost": dir_cost, "fail_risk": fail_risk,
                "cost": cost, "reward": reward, "discount": discount_factor
            }

        terminated = done

        return observation, reward, terminated, done, info

    def render(self):
        print(f"Step {self._time}: CS = {self._state}")

    def close(self):
        pass


if __name__ == '__main__':
    from element.problem_setup import ncs, na, gamma
    from element.problem_setup import action_model, unit_costs
    import matplotlib.pyplot as plt
    plt.ion()

    horizon = 200

    max_step = horizon
    max_cost = unit_costs.max()
    state_prob = np.zeros((ncs,))
    state_prob[0] = 1



    bridge= SingleElement(
        state_size=ncs, action_size=na,
        max_step=max_step,
        reset_prob=state_prob,
        action_model=action_model,
        unit_costs=unit_costs, 
        cost_params=(None, max_cost),
        discount=gamma,
        random_state='off',
    )

    # test env
    # test policy:
     # --------------------------------------------------------
    # 5- the followng piolicy is not aligned with the code 
    # if CS4 is greater than 30% or CS5 is greater than 5%, rehabilitate
    # elif CS4 is greater than 10%, repair
    # else do nothing
    states_log = []
    action_log = []
    reward_log = []

    states, _ = bridge.reset()
    for t in range(max_step):
        if states[3] > 0.3 or states[4] > 0.05:
            action = 3
        elif states[3] > 0.1:
            action = 2
        else:
            action = 0
        next_states, reward, terminated, done, info = \
            bridge.step(action)
        states_log.append(states)
        action_log.append(action)
        reward_log.append(reward)
        states = next_states


    for i in range(ncs):
        fig, ax = plt.subplots(1,1) 
        t = range(max_step)
        cs = [s[i] for s in states_log]
        ax.plot(t, cs)
        ax.set_title(f'Condition State {i+1} over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'CS {i+1}(Prob)')
    
    fig, ax = plt.subplots(1,1)
    t = range(max_step)
    ax.plot(t, action_log, 'o-')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Taken')

    fig, ax = plt.subplots(1,1)
    t = range(max_step)
    ax.plot(t, reward_log, '^-')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')

# %%
