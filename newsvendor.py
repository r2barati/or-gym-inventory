'''
Example taken from Balaji et al.
Paper: https://arxiv.org/abs/1911.10641
GitHub: https://github.com/awslabs/or-rl-benchmarks

Modified for Gymnasium compatibility.
'''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

class NewsvendorEnv(gym.Env):
    '''
    Multi-Period Newsvendor with Lead Times (Gymnasium Compatible)

    The MPNV requires meeting stochastic demand by having sufficient
    inventory on hand to satisfy customers. The inventory orders are not
    instantaneous and have multi-period lead times. Additionally, there are
    costs associated with holding unsold inventory, however unsold inventory
    expires at the end of each period.

    Observation:
        Type: Box
        State Vector: S = (p, c, h, k, mu, x_l, ..., x_1)
        p = price
        c = cost
        h = holding cost per unit
        k = lost sales penalty per unit
        mu = mean of demand distribution (Poisson)
        x_i = quantity ordered i periods ago, arriving in (lead_time - i + 1) periods.
              x_l is the order placed l periods ago (arriving next), x_1 is the order placed 1 period ago.
              In the code state[5] is arriving next, state[5 + lead_time - 1] is the most recent order.

    Actions:
        Type: Box
        Amount of product to order (non-negative float).

    Reward:
        Profit for the current period:
        Revenue - Purchase Cost (of arriving inventory) - Holding Cost - Lost Sales Penalty

    Initial State:
        Randomized parameters p, c, h, k, and mu, with no inventory in the pipeline.

    Episode Termination/Truncation:
        The episode is truncated when the step_limit is reached. There is no
        natural termination condition defined within the environment logic.
    '''
    metadata = {"render_modes": [], "render_fps": 4} # Add metadata if needed

    def __init__(self,
                 lead_time: int = 5,
                 max_inventory: int = 4000,
                 max_order_quantity: int = 2000,
                 step_limit: int = 40,
                 p_max: float = 100.0,
                 h_max: float = 5.0,
                 k_max: float = 10.0,
                 mu_max: float = 200.0,
                 gamma: float = 1.0): # Discount factor (often handled by agent, but can be part of env dynamics if needed)
        super().__init__() # Initialize the parent class

        # Environment Parameters
        self.lead_time = max(0, lead_time) # Ensure lead_time is non-negative
        self.max_inventory = max_inventory
        self.max_order_quantity = max_order_quantity
        self.step_limit = step_limit
        self.p_max = p_max
        self.h_max = h_max
        self.k_max = k_max
        self.mu_max = mu_max
        self.gamma = gamma # Note: Discounting is usually part of the RL algorithm, not the env reward calc, but kept as per original example

        # Define Observation Space
        self.obs_dim = self.lead_time + 5
        obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        obs_high = np.array(
            [self.p_max, self.p_max, self.h_max, self.k_max, self.mu_max] +
            [self.max_order_quantity] * self.lead_time,
            dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Define Action Space
        self.action_space = spaces.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([self.max_order_quantity], dtype=np.float32),
            dtype=np.float32)

        # Internal state variables (initialized in reset)
        self.state = None
        self.step_count = 0
        self.price = 0.0
        self.cost = 0.0
        self.h = 0.0
        self.k = 0.0
        self.mu = 0.0


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Important: Call parent reset for seeding RNG

        # Randomize costs using the environment's seeded RNG
        self.price = max(1, self.np_random.random() * self.p_max)
        # Ensure cost is less than or equal to price
        self.cost = max(1, self.np_random.random() * self.price)
        # Ensure holding cost is less than or equal to cost
        self.h = self.np_random.random() * min(self.cost, self.h_max)
        self.k = self.np_random.random() * self.k_max
        self.mu = self.np_random.random() * self.mu_max

        # Initialize state: [price, cost, h, k, mu, inventory_pipeline...]
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.state[:5] = np.array([self.price, self.cost, self.h, self.k, self.mu], dtype=np.float32)
        # Pipeline state (state[5:]) is already zeros (no incoming inventory)

        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs one time step of the environment's dynamics."""
        self.step_count += 1

        # --- Action ---
        # Ensure action is a scalar float/int
        order_qty_raw = action.item() if isinstance(action, np.ndarray) and action.size == 1 else action
        order_qty = np.clip(order_qty_raw, 0, self.max_order_quantity) # Ensure non-negative and below max order

        # Cap the order before a zero-lead-time delivery can be sold.
        current_pipeline_inventory = self.state[5:].sum()
        order_qty = max(0, min(order_qty, self.max_inventory - current_pipeline_inventory))
        if self.lead_time > 0:
            inv_on_hand = self.state[5] # Inventory arriving this period
        else: # No lead time means the capped order is available instantly
            inv_on_hand = order_qty

        # --- Dynamics ---
        demand = self.np_random.poisson(self.mu) # Use seeded RNG

        # Calculate sales and costs
        sales_units = min(inv_on_hand, demand)
        revenue = sales_units * self.price

        excess_inventory = max(0, inv_on_hand - demand)
        short_inventory = max(0, demand - inv_on_hand)

        # Costs Calculation - Revisit based on exact definition needed
        # Original code's purchase_cost seemed potentially off (multiplied by excess_inventory and order_qty).
        # A more standard interpretation might be:
        # 1. Cost of inventory *arriving* this period (inv_on_hand): inv_on_hand * self.cost
        # 2. Cost of inventory *ordered* this period (order_qty), perhaps discounted: order_qty * self.cost * self.gamma**self.lead_time
        # Let's use interpretation #2 as it aligns better with the original code's discounting idea.
        # If cost is incurred upon arrival, remove the gamma term and use inv_on_hand.
        purchase_cost = order_qty * self.cost # Cost incurred now for order placed now (arriving later)
                                               # If cost is paid on *arrival*, this needs rethinking. Assuming paid on *order*.
                                               # No discounting applied here directly, agent handles discounting future rewards.

        holding_cost = excess_inventory * self.h
        lost_sales_penalty = short_inventory * self.k

        # Reward: Revenue - Costs
        reward = revenue - purchase_cost - holding_cost - lost_sales_penalty

        # --- State Update ---
        # Update inventory pipeline: shift orders left, add new order
        new_pipeline = np.zeros(self.lead_time, dtype=np.float32)
        if self.lead_time > 0:
            # Shift existing pipeline orders one step closer to arrival
            new_pipeline[:-1] = self.state[6 : 5 + self.lead_time] # Indices 6 to end
            # Add the new order at the end of the pipeline
            new_pipeline[-1] = order_qty
        # Else (lead_time == 0): pipeline remains empty, order was instantly available.

        # Update the full state vector
        self.state[5:] = new_pipeline

        # Ensure state remains correct type
        self.state = self.state.astype(np.float32)

        # --- Termination/Truncation ---
        terminated = False # No natural end state in this environment definition
        truncated = self.step_count >= self.step_limit # Check if step limit is reached

        # --- Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info()
        info['demand'] = demand # Add potentially useful debug info
        info['revenue'] = revenue
        info['purchase_cost'] = purchase_cost
        info['holding_cost'] = holding_cost
        info['lost_sales_penalty'] = lost_sales_penalty


        # Return Gymnasium 5-tuple: obs, reward, terminated, truncated, info
        # Ensure reward is a standard float
        return observation, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Returns the current observation."""
        return self.state.copy()

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information."""
        # Include parameters that don't change mid-episode but define the instance
        return {
            "price": self.price,
            "cost": self.cost,
            "holding_cost_rate": self.h,
            "penalty_cost_rate": self.k,
            "demand_mean": self.mu,
            "lead_time": self.lead_time,
            "step_count": self.step_count
        }

    def render(self):
        """(Optional) Renders the environment."""
        # This environment doesn't have a visual representation.
        pass # Or print state information if desired for debugging

    def close(self):
        """(Optional) Performs any necessary cleanup."""
        pass # Nothing specific needed here

# --- Example Usage ---
if __name__ == '__main__':
    print("Testing NewsvendorEnv with Gymnasium...")

    # Instantiate the environment
    env = NewsvendorEnv(lead_time=5, step_limit=50)

    # Reset the environment to get the initial observation and info
    observation, info = env.reset(seed=42) # Use a seed for reproducibility
    print("Initial Observation:", observation)
    print("Initial Info:", info)
    print("-" * 20)

    total_reward = 0
    terminated = False
    truncated = False
    step = 0

    # Run an episode with random actions
    while not terminated and not truncated:
        step += 1
        # Sample a random action from the action space
        action = env.action_space.sample()
        # print(f"Step: {step}, Action: {action[0]:.2f}")

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # print(f"  Observation: {observation}")
        # print(f"  Reward: {reward:.2f}")
        # print(f"  Terminated: {terminated}, Truncated: {truncated}")
        # print(f"  Info: {info}")
        # print("-" * 10)

    print("=" * 20)
    print(f"Episode finished after {step} steps.")
    print(f"Total Reward: {total_reward:.2f}")
    print("Final Observation:", observation)
    print("Final Info:", info)

    env.close()