'''
Multi-period inventory management
Original by: Hector Perez, Christian Hubbs, Owais Sarwar (OR-Gym)
Modified for Gymnasium compatibility.
4/14/2020, updated 2024
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import poisson, binom, randint, geom # Keep scipy for distribution parameters if needed, but use np_random for sampling
from typing import Optional, Tuple, Dict, Any, List

# Helper function to handle environment configuration (alternative to or_gym's assign_env_config)
def assign_env_config(self, config: Dict[str, Any]):
    for key, value in config.items():
        setattr(self, key, value)

class InvManagementMasterEnv(gym.Env):
    '''
    Gymnasium-compatible Multi-Period Multi-Echelon Inventory Management Environment.

    Simulates a supply chain with multiple stages (echelons). Each stage holds inventory
    and potentially produces material for the downstream stage. Material flows down
    the chain with lead times. Customer demand occurs at the retailer (stage 0).
    The agent decides replenishment order quantities for stages 0 to M-1.

    For detailed explanation of events, see original docstring or OR-Gym documentation.

    Observation Space:
        Box space representing the inventory position at each stage, potentially including
        pipeline inventory information depending on the state representation chosen.
        The default state includes on-hand inventory at each stage [0, M-1] followed
        by the actions (orders) placed in the previous `lt_max` periods, flattened.
        Shape: (num_stages - 1) * (lead_time_max + 1)

    Action Space:
        Box space representing the non-negative integer replenishment quantities for
        each stage [0, M-1].
        Shape: (num_stages - 1,)

    Reward:
        Discounted profit for the current period.
        Profit = Revenue - Procurement Costs - Holding Costs - Penalty Costs (Backlog/Lost Sales)
    '''
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self,
                 periods: int = 30,
                 I0: List[int] = [100, 100, 200],
                 p: float = 2,
                 r: List[float] = [1.5, 1.0, 0.75, 0.5],
                 k: List[float] = [0.10, 0.075, 0.05, 0.025],
                 h: List[float] = [0.15, 0.10, 0.05],
                 c: List[int] = [100, 90, 80],
                 L: List[int] = [3, 5, 10],
                 backlog: bool = True,
                 dist: int = 1,
                 dist_param: Dict = {'mu': 20},
                 alpha: float = 0.97,
                 seed_int: int = 0, # Note: Seeding now primarily done via reset()
                 user_D: Optional[List[int]] = None,
                 env_config: Optional[Dict] = None):
        super().__init__() # Initialize the base class

        # --- Default Parameters ---
        self.periods = periods
        self.I0 = I0
        self.p = p
        self.r = r
        self.k = k
        self.h = h
        self.c = c
        self.L = L
        self.backlog = backlog
        self.dist = dist
        self.dist_param = dist_param
        self.alpha = alpha
        self.seed_int = seed_int # Keep for potential initial seeding if needed
        self.user_D = user_D if user_D is not None else []

        # --- Apply custom configuration ---
        if env_config:
            assign_env_config(self, env_config) # Apply overrides

        # --- Process Parameters ---
        self.init_inv = np.array(list(self.I0), dtype=np.int32)
        self.num_periods = self.periods
        self.unit_price = np.append(self.p, self.r[:-1]).astype(np.float32) # cost to stage i is price to stage i+1
        self.unit_cost = np.array(self.r, dtype=np.float32)
        self.demand_cost = np.array(self.k, dtype=np.float32)
        self.holding_cost = np.append(self.h, 0).astype(np.float32) # holding cost at last stage is 0
        self.supply_capacity = np.array(list(self.c), dtype=np.int64) # Use int64 for capacities/orders
        self.lead_time = np.array(list(self.L), dtype=np.int64)
        self.discount = self.alpha
        self.user_D = np.array(list(self.user_D), dtype=np.int64)

        self.num_stages = len(self.init_inv) + 1
        m = self.num_stages
        self.lt_max = 0 if m <= 1 else int(self.lead_time.max()) # Ensure lt_max is int

        # --- Input Validation ---
        self._validate_inputs()

        # --- Demand Distribution Setup ---
        self._setup_demand_distribution()

        # --- Define Spaces ---
        # Action space (reorder quantities for each stage; non-negative integers)
        # Shape: (m-1,) where m is num_stages
        self.action_space = spaces.Box(
            low=np.zeros(m - 1, dtype=np.int64),
            high=self.supply_capacity.astype(np.int64), # Max possible order is capacity
            shape=(m - 1,),
            dtype=np.int64) # Actions are discrete counts

        # Observation space (On-hand inventory + pipeline inventory representation)
        # Shape: (m-1) * (lt_max + 1)
        self.pipeline_length = (m - 1) * (self.lt_max + 1)
        # Define reasonable bounds - these might need tuning based on typical values
        inv_capacity_sum = self.supply_capacity.sum() * self.num_periods * 2 # Heuristic upper bound
        obs_low = -np.ones(self.pipeline_length, dtype=np.int64) * inv_capacity_sum if self.backlog else np.zeros(self.pipeline_length, dtype=np.int64)
        obs_high = np.ones(self.pipeline_length, dtype=np.int64) * inv_capacity_sum
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.pipeline_length,),
            dtype=np.int64) # Observations are counts

        # Internal state variables will be initialized in reset()
        self.period: int = 0
        self.state: np.ndarray = None
        self.I: np.ndarray = None # On-hand inventory
        self.T: np.ndarray = None # Pipeline inventory (conceptual, calculated in state)
        self.action_log: np.ndarray = None # Log of past actions (orders)
        self.D: np.ndarray = None # Demand history
        self.S: np.ndarray = None # Sales history
        self.B: np.ndarray = None # Backlog history
        self.LS: np.ndarray = None # Lost Sales history
        self.P: np.ndarray = None # Profit history
        self.R: np.ndarray = None # Replenishment order history


    def _validate_inputs(self):
        """ Perform input validation checks. """
        m = self.num_stages
        assert np.all(self.init_inv >= 0), "Initial inventory cannot be negative"
        assert self.num_periods > 0, "Number of periods must be positive"
        assert np.all(self.unit_price >= 0), "Sales prices cannot be negative"
        assert np.all(self.unit_cost >= 0), "Procurement costs cannot be negative"
        assert np.all(self.demand_cost >= 0), "Unfulfilled demand costs cannot be negative"
        assert np.all(self.holding_cost >= 0), "Holding costs cannot be negative"
        assert np.all(self.supply_capacity > 0), "Supply capacities must be positive"
        assert np.all(self.lead_time >= 0), "Lead times cannot be negative"
        assert isinstance(self.backlog, bool), "Backlog parameter must be boolean"
        assert m >= 2, "Minimum number of stages is 2"
        assert len(self.unit_cost) == m, f"Length of r ({len(self.unit_cost)}) != num stages ({m})"
        assert len(self.demand_cost) == m, f"Length of k ({len(self.demand_cost)}) != num stages ({m})"
        # Note: holding_cost includes dummy 0 for last stage, so len == m
        assert len(self.holding_cost) == m, f"Length of h ({len(self.holding_cost)}) != num stages ({m})"
        assert len(self.supply_capacity) == m - 1, f"Length of c ({len(self.supply_capacity)}) != num stages - 1 ({m-1})"
        assert len(self.lead_time) == m - 1, f"Length of L ({len(self.lead_time)}) != num stages - 1 ({m-1})"
        assert self.dist in [1, 2, 3, 4, 5], "dist must be one of 1, 2, 3, 4, 5"
        if self.dist == 5:
             assert len(self.user_D) == self.num_periods, "User specified demand length != num periods"
        # Scipy validation is complex, rely on runtime errors for now or add specific checks if needed
        assert 0 < self.alpha <= 1, "alpha must be in the range (0, 1]"

    def _setup_demand_distribution(self):
        """ Sets up the demand distribution function based on self.dist """
        if self.dist == 1: # Poisson
            self._sample_demand = lambda: self.np_random.poisson(lam=self.dist_param['mu'])
        elif self.dist == 2: # Binomial
            self._sample_demand = lambda: self.np_random.binomial(n=self.dist_param['n'], p=self.dist_param['p'])
        elif self.dist == 3: # Randint (Uniform Integer)
            # np_random.integers is exclusive of high, so add 1
            self._sample_demand = lambda: self.np_random.integers(low=self.dist_param['low'], high=self.dist_param['high'] + 1)
        elif self.dist == 4: # Geometric
             # Note: numpy's geometric definition might differ slightly from scipy's in terms of trials vs failures. Check definition if precise matching is needed.
            self._sample_demand = lambda: self.np_random.geometric(p=self.dist_param['p'])
        elif self.dist == 5: # User defined
            self._sample_demand = lambda: self.user_D[self.period] if self.period < len(self.user_D) else 0
        else:
            raise ValueError(f"Invalid distribution choice: {self.dist}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Args:
            seed: The seed for the random number generator.
            options: Optional dictionary with additional reset options (unused here).

        Returns:
            A tuple containing the initial observation and an information dictionary.
        """
        super().reset(seed=seed) # Important: seeds self.np_random

        periods = self.num_periods
        m = self.num_stages

        # Initialize state arrays
        self.I = np.zeros((periods + 1, m - 1), dtype=np.int64) # On-hand inventory
        self.T = np.zeros((periods + 1, m - 1), dtype=np.int64) # Conceptual pipeline inventory start
        self.R = np.zeros((periods, m - 1), dtype=np.int64)   # Replenishment orders placed
        self.D = np.zeros(periods, dtype=np.int64)            # Customer demand history
        self.S = np.zeros((periods, m), dtype=np.int64)       # Sales / fulfilled orders history
        self.B = np.zeros((periods + 1, m), dtype=np.int64)   # Backlog history (extra slot for initial)
        self.LS = np.zeros((periods, m), dtype=np.int64)      # Lost sales history
        self.P = np.zeros(periods, dtype=np.float32)          # Profit history
        self.action_log = np.zeros((periods, m - 1), dtype=np.int64) # Log actions for state calculation

        # Initialization
        self.period = 0                     # Reset time
        self.I[0, :] = self.init_inv.astype(np.int64) # Initial inventory
        # T[0,:], B[0,:], action_log[0] are implicitly zeros

        # Set initial state and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advances the environment by one time step.

        Args:
            action: The replenishment order quantities for stages 0 to M-1.

        Returns:
            A tuple containing:
                - observation: The next state observation.
                - reward: The profit obtained in this step.
                - terminated: Whether the episode ended naturally (always False here).
                - truncated: Whether the episode ended due to time limit (True if period >= num_periods).
                - info: Auxiliary information dictionary.
        """
        t = self.period
        m = self.num_stages
        m_minus_1 = m - 1 # Stages with inventory/orders {0, ..., M-2}
        L = self.lead_time
        c = self.supply_capacity

        # --- 0) Stages place replenishment orders ---
        # Enforce the public action-space contract before recording orders.
        action = np.asarray(action)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Action quantities must be finite")
        R_requested = np.clip(
            action, self.action_space.low, self.action_space.high
        ).astype(np.int64)

        # Add previous period's backlog for replenishment orders (stages 1 to m)
        current_order_request = R_requested.copy()
        if t >= 1:
            current_order_request += self.B[t, 1:] # Backlog from period t (calculated at end of t-1)

        # Determine actual fulfilled replenishment order R[t] based on supplier capacity and inventory
        # Available inventory at the supplier stages (m_supplier = m_order + 1)
        # Stage m-1 (last production) has infinite raw materials
        supplier_inv = np.append(self.I[t, 1:], np.inf) # Inventory at stages 1 to m-1, plus Inf for stage m

        # Enforce capacity constraints
        R_fulfill = np.minimum(current_order_request, c)
        # Enforce supplier inventory constraints
        R_fulfill = np.minimum(R_fulfill, supplier_inv).astype(np.int64)

        self.R[t, :] = R_fulfill # Store actual replenishment order fulfilled
        self.action_log[t, :] = R_requested # Log the originally requested action for state calculation

        # --- 1) Receive incoming inventory ---
        I_current = self.I[t, :].copy() # On-hand inventory at start of period t
        received_shipments = np.zeros(m_minus_1, dtype=np.int64)
        for i in range(m_minus_1):
            if t - L[i] >= 0:
                shipment_due = self.R[t - L[i], i]
                I_current[i] += shipment_due
                received_shipments[i] = shipment_due

        # --- 2) Customer demand occurs ---
        demand_realized = max(0, self._sample_demand()) # Ensure demand is non-negative
        self.D[t] = demand_realized

        # --- 3) Fill demand ---
        demand_to_fill = demand_realized
        if t >= 1:
            demand_to_fill += self.B[t, 0] # Add backlog from previous period for stage 0

        sales_stage0 = min(I_current[0], demand_to_fill)
        I_current[0] -= sales_stage0 # Update inventory after sales

        # --- 4 & 5 combined: Determine sales at other stages, backlog/lost sales, holding costs ---
        S_current = np.zeros(m, dtype=np.int64) # Sales array for period t [S0, S1, ..., Sm-1]
        U_current = np.zeros(m, dtype=np.int64) # Unfulfilled orders [U0, U1, ..., Um-1]
        S_current[0] = sales_stage0
        S_current[1:] = R_fulfill # Sales from stage i+1 to stage i are the fulfilled orders R[i]

        self.S[t, :] = S_current

        # Each internal supplier (stages 1..m-2) ships to the stage immediately below it.
        # R_fulfill[i] is shipped by stage i+1, so debit R_fulfill[:-1].
        I_current[1:] -= R_fulfill[:-1]

        # Calculate unfulfilled orders (U)
        U_current[0] = demand_to_fill - sales_stage0
        U_current[1:] = current_order_request - R_fulfill

        # Calculate backlog (B) or lost sales (LS)
        if self.backlog:
            self.B[t + 1, :] = U_current # Backlog at end of period t (start of t+1)
            self.LS[t, :] = 0
        else:
            self.LS[t, :] = U_current # Lost sales occur in period t
            self.B[t + 1, :] = 0

        # --- Calculate period profit/reward ---
        revenue = self.unit_price * S_current
        procurement_cost = self.unit_cost * S_current # Cost is paid by the receiving stage for units *sold* by supplier
                                                       # Note: S_current[m-1] is production at last stage
        holding_cost = self.holding_cost * np.maximum(0, np.append(I_current, 0)) # Cost on ending inventory (positive only)
        penalty_cost = self.demand_cost * U_current    # Cost for unfulfilled demand/orders

        period_profit = np.sum(revenue - procurement_cost - holding_cost - penalty_cost)
        discounted_profit = (self.discount ** t) * period_profit
        self.P[t] = discounted_profit

        # --- Update state for next period ---
        self.I[t + 1, :] = I_current # Store ending inventory for start of t+1
        # Pipeline inventory T is implicitly captured by the state representation

        # --- Prepare return values ---
        self.period += 1
        observation = self._get_obs()
        info = self._get_info()
        # Add more details to info if needed for debugging
        info.update({
            'period_profit': period_profit,
            'revenue': revenue.sum(),
            'procurement_cost': procurement_cost.sum(),
            'holding_cost': holding_cost.sum(),
            'penalty_cost': penalty_cost.sum(),
            'demand_realized': demand_realized,
            'sales': S_current,
            'unfulfilled': U_current,
            'ending_inventory': I_current,
            'backlog_start_of_next': self.B[t + 1, :]
        })


        reward = float(discounted_profit) # Ensure reward is float
        terminated = False # This environment doesn't have a natural termination condition
        truncated = self.period >= self.num_periods # Truncated if max periods reached

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation array based on current on-hand inventory
        and recent actions (orders placed).

        State: [I[t,0], ..., I[t,M-2],        # Current on-hand inventory
                Action[t-1, 0], ..., Action[t-1, M-2], # Orders placed last period
                ...,
                Action[t-lt_max, 0], ..., Action[t-lt_max, M-2] # Orders placed lt_max periods ago
               ]
        Flattened into a 1D array.
        """
        t = self.period
        m_minus_1 = self.num_stages - 1
        lt_max = self.lt_max

        state = np.zeros(self.pipeline_length, dtype=np.int64)

        # Current on-hand inventory (at start of period t)
        state[:m_minus_1] = self.I[t, :m_minus_1]

        # Pipeline inventory represented by past actions (orders placed)
        if t > 0:
            # Determine how many past periods of actions to include (up to lt_max)
            num_past_periods = min(t, lt_max)
            # Most recent action first, matching the documented observation layout.
            # Pad the oldest (not yet observed) history slots with zeros.
            past_actions = self.action_log[t - num_past_periods : t, :][::-1]
            state[m_minus_1 : m_minus_1 + past_actions.size] = past_actions.flatten()

        # Add backlog to the state? Original code included it in inventory position.
        # This state definition separates on-hand from pipeline/orders.
        # If backlog is needed in obs, the observation space and calculation needs adjustment.
        # For now, keeping backlog out of the observation, but it affects dynamics.

        # Ensure dtype matches observation space
        return state.astype(self.observation_space.dtype)


    def _get_info(self) -> Dict[str, Any]:
        """ Returns auxiliary information about the current state. """
        return {
            "period": self.period,
            "current_inventory_on_hand": self.I[self.period].copy(),
            "current_backlog": self.B[self.period].copy() # Backlog at start of current period
        }

    def sample_action(self) -> np.ndarray:
        """ Samples a random action from the action space. """
        return self.action_space.sample()

    def render(self):
        """ Implement visualization if needed. """
        # Example: print current status
        print(f"Period: {self.period}")
        print(f"  Inventory (On-Hand): {self.I[self.period]}")
        print(f"  Backlog (Start of Period): {self.B[self.period]}")
        if self.period > 0:
             print(f"  Demand (Previous): {self.D[self.period-1]}")
             print(f"  Sales (Previous): {self.S[self.period-1]}")
             print(f"  Profit (Previous): {self.P[self.period-1]:.2f}")


    def close(self):
        """ Perform any necessary cleanup. """
        pass

    # --- Optional: Keep helper methods if needed externally ---
    # def base_stock_action(self,z): ... (needs update for Gymnasium state/reward)
    # def _update_base_stock_policy_state(self): ... (needs update)


# --- Subclasses for Backlog vs Lost Sales ---

class InvManagementBacklogEnv(InvManagementMasterEnv):
    """ Inventory management environment with backlogging enabled (default). """
    def __init__(self, *args, **kwargs):
        # The backlog variant must remain a backlog environment even when
        # callers supply conflicting env_config settings.
        kwargs['backlog'] = True
        if kwargs.get('env_config') is not None:
            kwargs['env_config'] = {**kwargs['env_config'], 'backlog': True}
        super().__init__(*args, **kwargs)

class InvManagementLostSalesEnv(InvManagementMasterEnv):
    """ Inventory management environment with lost sales (backlog=False). """
    def __init__(self, *args, **kwargs):
        # Force lost-sales semantics before the parent builds its spaces and
        # dynamics; copy config to avoid mutating the caller's dictionary.
        kwargs['backlog'] = False
        if kwargs.get('env_config') is not None:
            kwargs['env_config'] = {**kwargs['env_config'], 'backlog': False}
        super().__init__(*args, **kwargs)
        # Observation space lower bound is 0 for lost sales env
        m_minus_1 = self.num_stages - 1
        lt_max = self.lt_max
        pipeline_length = (m_minus_1) * (lt_max + 1)
        inv_capacity_sum = self.supply_capacity.sum() * self.num_periods * 2 # Heuristic upper bound
        self.observation_space = spaces.Box(
            low=np.zeros(pipeline_length, dtype=np.int64), # Lower bound is 0
            high=np.ones(pipeline_length, dtype=np.int64) * inv_capacity_sum,
            shape=(pipeline_length,),
            dtype=np.int64)


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing Gymnasium-compatible InvManagement Environments...")

    # Test Backlog Env
    print("\n--- Testing Backlog Environment ---")
    # Use smaller parameters for quick testing
    config_backlog = {
        'periods': 10,
        'I0': [10, 10],
        'p': 5,
        'r': [3, 2, 1],
        'k': [1, 1, 1],
        'h': [0.5, 0.2],
        'c': [15, 20],
        'L': [1, 2],
        'dist_param': {'mu': 8}
    }
    env_backlog = InvManagementBacklogEnv(env_config=config_backlog)

    obs, info = env_backlog.reset(seed=42)
    print(f"Initial Obs (Backlog): {obs}")
    print(f"Initial Info (Backlog): {info}")

    total_reward_backlog = 0
    terminated = False
    truncated = False

    for _ in range(env_backlog.num_periods):
        action = env_backlog.sample_action() # Sample random action
        # print(f"Action: {action}")
        obs, reward, terminated, truncated, info = env_backlog.step(action)
        total_reward_backlog += reward
        # env_backlog.render() # Uncomment to see step details
        if terminated or truncated:
            break

    print(f"\nEpisode Finished (Backlog). Total Reward: {total_reward_backlog:.2f}")
    print(f"Final Info (Backlog): {info}")
    env_backlog.close()

    # Test Lost Sales Env
    print("\n--- Testing Lost Sales Environment ---")
    config_lost_sales = config_backlog.copy() # Start with same config
    env_lost_sales = InvManagementLostSalesEnv(env_config=config_lost_sales)

    obs, info = env_lost_sales.reset(seed=123)
    print(f"Initial Obs (Lost Sales): {obs}")
    print(f"Initial Info (Lost Sales): {info}")

    total_reward_lost_sales = 0
    terminated = False
    truncated = False

    for _ in range(env_lost_sales.num_periods):
        action = env_lost_sales.sample_action() # Sample random action
        obs, reward, terminated, truncated, info = env_lost_sales.step(action)
        total_reward_lost_sales += reward
        if terminated or truncated:
            break

    print(f"\nEpisode Finished (Lost Sales). Total Reward: {total_reward_lost_sales:.2f}")
    print(f"Final Info (Lost Sales): {info}")
    env_lost_sales.close()

    print("\nTesting complete.")
