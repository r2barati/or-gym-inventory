'''
Network inventory management environment.
Original by: Hector Perez, Christian Hubbs, Can Li (OR-Gym)
Modified for Gymnasium compatibility.
9/14/2020, updated 2024
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import poisson, binom, randint, geom # Keep for defining distributions
from typing import Optional, Tuple, Dict, Any, List, Union

# Helper function to handle environment configuration (alternative to or_gym's assign_env_config)
def assign_env_config(self, config: Dict[str, Any]):
    for key, value in config.items():
        # Special handling for graph if provided in config
        if key == 'graph' and isinstance(value, nx.DiGraph):
             # print("Assigning graph from config") # Debug print
             self.graph = value.copy() # Use a copy
        else:
            setattr(self, key, value)

class NetInvMgmtMasterEnv(gym.Env):
    '''
    Gymnasium-compatible Multi-Period Multi-Node Network Inventory Management Environment.

    Simulates a supply network with production, distribution, raw material, and market nodes.
    Manages inventory, production, lead times, costs, and demand across the network.
    The agent decides replenishment order quantities between connected nodes.

    See original docstring or OR-Gym documentation for detailed event sequence.

    Observation Space:
        Box space representing the system state, typically including:
        - Market demands/backlog at the start of the period for each (retailer, market) link.
        - On-hand inventory at each main node (distribution/factory).
        - Pipeline inventory, represented by orders placed in previous periods for each link with a lead time.
        Flattened into a 1D array.

    Action Space:
        Box space representing the non-negative replenishment quantities requested
        for each link that allows reordering (i.e., not market links).
        Shape: (number_of_reorder_links,)

    Reward:
        Total discounted profit across all nodes for the current period.
        Profit considers sales revenue, purchasing costs, operating costs, holding costs (on-hand and pipeline),
        and penalty costs (backlog/lost sales).
    '''
    metadata = {"render_modes": ["human"], "render_fps": 4} # Added "human" for plot_network

    def __init__(self,
                 graph: Optional[nx.DiGraph] = None, # Allow passing a pre-defined graph
                 num_periods: int = 30,
                 backlog: bool = True,
                 alpha: float = 1.00,
                 seed_int: int = 0,
                 user_D: Optional[Dict[Tuple[int, int], List[int]]] = None,
                 sample_path: Optional[Dict[Tuple[int, int], bool]] = None,
                 env_config: Optional[Dict] = None):
        super().__init__() # Initialize the base class

        # --- Default Parameters ---
        self.num_periods = num_periods
        self.backlog = backlog
        self.alpha = alpha
        self.seed_int = seed_int # Keep for potential initial seeding
        self.user_D = user_D if user_D is not None else {}
        self.sample_path = sample_path if sample_path is not None else {}

        # --- Graph Definition ---
        # Use provided graph or create default if None
        if graph is not None:
            self.graph = graph.copy() # Use a copy
        else:
            self._create_default_graph() # Define default structure if none provided

        # --- Apply custom configuration AFTER graph is set ---
        # Need to handle env_config slightly differently now backlog is set above
        _env_config = dict(env_config) if env_config else {}
        _env_config['backlog'] = self.backlog # Constructor's backlog choice overrides config
        assign_env_config(self, _env_config) # Apply overrides from dict

        # --- Post-Graph Setup and Validation ---
        self._initialize_graph_dependent_attributes()
        self._validate_inputs()
        # Demand distributions need np_random, setup in reset first time
        # Call _setup_demand_distributions within reset after super().reset()

        # --- Define Spaces ---
        self._define_spaces()

        # Internal state variables will be initialized in reset()
        self.period: int = 0
        self.state: np.ndarray = None
        self.X: pd.DataFrame = None # On-hand inventory
        self.Y: pd.DataFrame = None # Pipeline inventory
        self.R: pd.DataFrame = None # Replenishment orders placed
        self.S: pd.DataFrame = None # Sales / fulfilled orders
        self.D: pd.DataFrame = None # Demand history
        self.U: pd.DataFrame = None # Unfulfilled demand/orders
        self.P: pd.DataFrame = None # Profit history
        # self.action_log: np.ndarray = None # Log actions (can be useful but not currently needed for state)

    def _create_default_graph(self):
        """ Creates the default graph structure if none is provided. """
        self.graph = nx.DiGraph()
        # Define nodes with attributes
        self.graph.add_nodes_from([0]) # Market
        self.graph.add_nodes_from([1], I0=100, h=0.030) # Retailer
        self.graph.add_nodes_from([2], I0=110, h=0.020) # Distributor
        self.graph.add_nodes_from([3], I0=80, h=0.015) # Distributor
        self.graph.add_nodes_from([4], I0=400, C=90, o=0.010, v=1.000, h=0.012) # Manufacturer
        self.graph.add_nodes_from([5], I0=350, C=90, o=0.015, v=1.000, h=0.013) # Manufacturer
        self.graph.add_nodes_from([6], I0=380, C=80, o=0.012, v=1.000, h=0.011) # Manufacturer
        self.graph.add_nodes_from([7, 8]) # Raw materials
        # Define edges with attributes
        self.graph.add_edges_from([
             # Edge (1,0) connects Retailer 1 to Market 0
            (1, 0, {'p': 2.000, 'b': 0.100, # p: price charged by retailer, b: backlog/lost sale cost
                    # Function to sample demand using environment's RNG
                    'demand_dist_func': lambda **p: self.np_random.poisson(**p),
                    # Parameters for the demand function (numpy poisson uses 'lam')
                    'dist_param': {'lam': 20}}),
            (2, 1, {'L': 5, 'p': 1.500, 'g': 0.010}), # L: Lead Time, p: purchase cost for receiver, g: pipeline holding cost
            (3, 1, {'L': 3, 'p': 1.600, 'g': 0.015}),
            (4, 2, {'L': 8, 'p': 1.000, 'g': 0.008}),
            (4, 3, {'L': 10, 'p': 0.800, 'g': 0.006}),
            (5, 2, {'L': 9, 'p': 0.700, 'g': 0.005}),
            (6, 2, {'L': 11, 'p': 0.750, 'g': 0.007}),
            (6, 3, {'L': 12, 'p': 0.800, 'g': 0.004}),
            (7, 4, {'L': 0, 'p': 0.150, 'g': 0.000}), # L=0 means immediate transfer
            (7, 5, {'L': 1, 'p': 0.050, 'g': 0.005}),
            (8, 5, {'L': 2, 'p': 0.070, 'g': 0.002}),
            (8, 6, {'L': 0, 'p': 0.200, 'g': 0.000})
        ])
        # Set default user_D for the market link if not overridden
        if (1,0) not in self.user_D:
            self.user_D[(1,0)] = np.zeros(self.num_periods)
        if (1,0) not in self.sample_path:
             self.sample_path[(1,0)] = False

    def _initialize_graph_dependent_attributes(self):
        """ Initializes attributes derived from the network graph. """
        # Save user_D and sample_path to graph metadata if provided
        for link, d in self.user_D.items():
            if link in self.graph.edges:
                 # Ensure user_D is stored as a list or array
                self.graph.edges[link]['user_D'] = list(d) if not isinstance(d, (list, np.ndarray)) else d
                self.graph.edges[link]['sample_path'] = self.sample_path.get(link, False)
            else:
                 print(f"Warning: Link {link} from user_D not found in graph.")

        # Ensure all market links have user_D and sample_path attributes (even if default)
        self.retail_links = [e for e in self.graph.edges() if 'L' not in self.graph.edges[e]]
        for link in self.retail_links:
             if 'user_D' not in self.graph.edges[link]:
                 self.graph.edges[link]['user_D'] = np.zeros(self.num_periods)
             if 'sample_path' not in self.graph.edges[link]:
                 self.graph.edges[link]['sample_path'] = False


        self.num_nodes = self.graph.number_of_nodes()
        # Identify node types
        self.market = [j for j in self.graph.nodes() if not list(self.graph.successors(j))]
        self.rawmat = [j for j in self.graph.nodes() if not list(self.graph.predecessors(j))]
        self.factory = [j for j in self.graph.nodes() if 'C' in self.graph.nodes[j]]
        # Distributors are nodes with inventory but no capacity, excluding raw materials
        self.distrib = [j for j in self.graph.nodes() if 'I0' in self.graph.nodes[j] and 'C' not in self.graph.nodes[j] and j not in self.rawmat]
        # Retailers are distributors whose successors include market nodes
        self.retail = [j for j in self.distrib if any(s in self.market for s in self.graph.successors(j))]

        self.main_nodes = sorted(list(set(self.distrib + self.factory))) # Nodes that hold inventory and place/receive orders

        # Identify link types
        self.reorder_links = sorted([e for e in self.graph.edges() if 'L' in self.graph.edges[e]]) # Edges with lead times (orderable)
        # self.retail_links are defined above
        self.network_links = sorted([e for e in self.graph.edges()]) # All edges

        # Calculate max lead time and pipeline length for observation space
        self.lead_times = {e: self.graph.edges[e]['L'] for e in self.reorder_links}
        self.lt_max = max(self.lead_times.values()) if self.lead_times else 0

        # Calculate total length needed for pipeline state representation
        self.pipeline_obs_length = sum(self.lead_times.values()) # Sum of lead times for state padding

        self.obs_dim = len(self.retail_links) + len(self.main_nodes) + self.pipeline_obs_length

        # Store max values for potential use in space bounds (can be refined)
        self.init_inv_max = max((self.graph.nodes[j].get('I0', 0) for j in self.main_nodes), default=100)
        self.capacity_max = max((self.graph.nodes[j].get('C', 0) for j in self.factory), default=100)
        self.order_cap_heuristic = (self.init_inv_max + self.capacity_max * 5) # Heuristic bound

    def _validate_inputs(self):
        """ Perform input validation checks. """
        # Basic structure checks
        nodes = set(self.graph.nodes())
        node_types = set(self.market) | set(self.distrib) | set(self.factory) | set(self.rawmat)
        # Allow nodes to exist that aren't classified if graph is complex, but warn maybe?
        # assert nodes == node_types, f"Node classification mismatch. Nodes: {nodes}, Classified: {node_types}"
        if nodes != node_types:
             print(f"Warning: Some nodes not classified: {nodes - node_types}")

        # Node attribute checks
        for j in self.graph.nodes():
            attrs = self.graph.nodes[j]
            if j in self.main_nodes: # Nodes that should have I0
                assert 'I0' in attrs and attrs['I0'] >= 0, f"Node {j}: Invalid or missing I0>=0"
                assert 'h' in attrs and attrs['h'] >= 0, f"Node {j}: Invalid or missing h>=0"
            if j in self.factory:
                assert 'C' in attrs and attrs['C'] > 0, f"Node {j}: Invalid or missing C>0"
                assert 'o' in attrs and attrs['o'] >= 0, f"Node {j}: Invalid or missing o>=0"
                assert 'v' in attrs and 0 < attrs['v'] <= 1, f"Node {j}: Invalid or missing v in (0, 1]"

        # Edge attribute checks
        for u, v, attrs in self.graph.edges(data=True):
            edge = (u, v)
            if edge in self.reorder_links: # Links with lead times
                assert 'L' in attrs and attrs['L'] >= 0, f"Edge {edge}: Invalid or missing L>=0"
                assert 'p' in attrs and attrs['p'] >= 0, f"Edge {edge}: Invalid or missing p>=0"
                assert 'g' in attrs and attrs['g'] >= 0, f"Edge {edge}: Invalid or missing g>=0"
            if edge in self.retail_links: # Links to markets
                assert 'p' in attrs and attrs['p'] >= 0, f"Edge {edge}: Invalid or missing p>=0 (price)"
                assert 'b' in attrs and attrs['b'] >= 0, f"Edge {edge}: Invalid or missing b>=0 (backlog cost)"
                assert 'demand_dist_func' in attrs or 'user_D' in attrs, f"Edge {edge}: Missing demand source ('demand_dist_func' or 'user_D')"
                if 'demand_dist_func' in attrs:
                    assert 'dist_param' in attrs, f"Edge {edge}: Missing 'dist_param' for 'demand_dist_func'"
                # Check user_D length only if it's expected to be used (not sample_path=True)
                if 'user_D' in attrs and np.sum(attrs['user_D']) > 0 and not attrs.get('sample_path', False):
                     assert len(attrs['user_D']) == self.num_periods, f"Edge {edge}: user_D length {len(attrs['user_D'])} != num_periods {self.num_periods}"

        # Other parameters
        assert isinstance(self.backlog, bool), "backlog must be boolean"
        assert 0 < self.alpha <= 1, "alpha must be in (0, 1]"
        assert self.num_periods > 0, "num_periods must be positive"

    def _setup_demand_distributions(self):
        """
        Sets up the demand sampling functions for market links using self.np_random.
        This needs to be called after super().reset() has initialized self.np_random.
        Stores the sampling function directly in the edge data.
        """
        for u, v, data in self.graph.edges(data=True):
            edge = (u, v)
            if edge in self.retail_links:
                 # If user_D is specified AND sample_path is False, use user_D directly
                use_user_d = ('user_D' in data and np.sum(data['user_D']) > 0 and not data.get('sample_path', False))

                if use_user_d:
                    # Demand taken directly from user_D list in step function
                    # Ensure index doesn't go out of bounds
                    data['_sample_demand'] = lambda e=edge: self.graph.edges[e]['user_D'][min(self.period, len(self.graph.edges[e]['user_D'])-1)]
                # Otherwise, set up sampling function based on demand_dist_func
                elif 'demand_dist_func' in data and 'dist_param' in data:
                    params = data['dist_param']
                    dist_func = data['demand_dist_func']
                    # Create lambda that calls the stored function with params using self.np_random
                    # This lambda captures the specific `dist_func` and `params` for this edge
                    # Ensures demand is non-negative integer
                    data['_sample_demand'] = lambda f=dist_func, p=params: max(0, int(round(f(**p))))
                else:
                    # Default to zero demand if no valid source specified
                    print(f"Warning: No valid demand source for edge {edge}. Defaulting to 0.")
                    data['_sample_demand'] = lambda: 0


    def _define_spaces(self):
        """ Defines the action and observation spaces. """
        # Action Space: Order quantity for each reorder link
        num_reorder_links = len(self.reorder_links)
        # Use float32 for actions, often preferred by RL libraries, round internally
        # Bounds can be high, agents should learn reasonable values
        action_high = np.ones(num_reorder_links, dtype=np.float32) * self.order_cap_heuristic * 2
        self.action_space = spaces.Box(
            low=np.zeros(num_reorder_links, dtype=np.float32),
            high=action_high,
            shape=(num_reorder_links,),
            dtype=np.float32)

        # Observation Space: Concatenation of demands, inventories, pipeline
        # Use float32 for observations
        obs_high_heuristic = self.order_cap_heuristic * self.num_periods * 2 # Large positive bound
        obs_low_heuristic = 0.0 if not self.backlog else -obs_high_heuristic # Allow negative if backlogged

        obs_low = np.full(self.obs_dim, obs_low_heuristic, dtype=np.float32)
        obs_high = np.full(self.obs_dim, obs_high_heuristic, dtype=np.float32)

        # Refine bounds for demand/backlog part (first len(retail_links) elements)
        obs_low[0:len(self.retail_links)] = 0.0 # Demand/backlog usually non-negative

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Resets the environment to its initial state. """
        super().reset(seed=seed) # Important: seeds self.np_random

        # Re-setup demand distributions now that self.np_random is initialized
        self._setup_demand_distributions()

        T = self.num_periods
        J = len(self.main_nodes)
        RM = len(self.retail_links)
        PS = len(self.reorder_links)
        SL = len(self.network_links) # All links

        # Initialize Pandas DataFrames for tracking state
        self.X = pd.DataFrame(data=np.zeros([T + 1, J]), columns=self.main_nodes, dtype=np.float64) # Inventory
        self.Y = pd.DataFrame(data=np.zeros([T + 1, PS]), columns=pd.MultiIndex.from_tuples(self.reorder_links), dtype=np.float64) # Pipeline
        self.R = pd.DataFrame(data=np.zeros([T, PS]), columns=pd.MultiIndex.from_tuples(self.reorder_links), dtype=np.float64) # Replenishment orders
        self.S = pd.DataFrame(data=np.zeros([T, SL]), columns=pd.MultiIndex.from_tuples(self.network_links), dtype=np.float64) # Sales/Flow
        self.D = pd.DataFrame(data=np.zeros([T, RM]), columns=pd.MultiIndex.from_tuples(self.retail_links), dtype=np.float64) # Demand
        self.U = pd.DataFrame(data=np.zeros([T + 1, RM]), columns=pd.MultiIndex.from_tuples(self.retail_links), dtype=np.float64) # Unfulfilled Demand (backlog at start of next period)
        self.P = pd.DataFrame(data=np.zeros([T, J]), columns=self.main_nodes, dtype=np.float64) # Profit per node

        # Initialization
        self.period = 0
        for j in self.main_nodes:
            self.X.loc[0, j] = self.graph.nodes[j].get('I0', 0) # Initial inventory
        # Y, R, S, D, U, P start at 0 implicitly

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation array based on current backlog, inventory, and pipeline state.
        State format: [Current Backlog (U[t]), Current On-Hand Inventories (X[t]), Flattened Pipeline History (R[t-L:t-1])]
        """
        t = self.period

        # 1. Current period's starting backlog/unfulfilled demand (U[t])
        # U[t] represents the unfulfilled demand from the *end* of period t-1.
        demand_backlog_component = self.U.loc[t, self.retail_links].values.flatten()

        # 2. Current on-hand inventory at main nodes (X[t])
        inventory_component = self.X.loc[t, self.main_nodes].values.flatten()

        # 3. Pipeline inventory state
        # Represented by orders *placed* in previous periods, padded to match total pipeline length
        pipeline_components = []
        for edge in self.reorder_links:
            lead_time = self.lead_times[edge]
            if lead_time == 0:
                continue # No pipeline state for L=0

            # Get past orders for this link up to L periods ago
            # R rows are indexed by period t, R[t] is order placed *in* period t
            # We need orders placed from t-L to t-1 (which arrive from period t to t+L-1)
            start_period_idx = max(0, t - lead_time)
            end_period_idx = t # pandas slicing is inclusive start, exclusive end for loc index when using integers

            # Ensure indices are within the bounds of the R dataframe (0 to T-1)
            start_period_idx = min(start_period_idx, self.num_periods - 1)
            end_period_idx = min(end_period_idx, self.num_periods)

            # Get the relevant slice of orders from the R dataframe for this edge
            # Handle case where t < L
            if start_period_idx >= end_period_idx:
                 past_orders = np.array([], dtype=np.float64)
            else:
                 # Select rows based on index range and the specific edge column
                 past_orders = self.R.loc[start_period_idx : end_period_idx - 1, edge].values

            # Pad with zeros if t < lead_time or slice is empty
            padded_pipeline = np.zeros(lead_time, dtype=np.float32)
            if len(past_orders) > 0:
                 # Fill from the end (most recent order is last in relevant history for state)
                 padded_pipeline[-len(past_orders):] = past_orders

            pipeline_components.append(padded_pipeline)

        if pipeline_components:
            pipeline_state = np.concatenate(pipeline_components)
        else:
            # Handle case with no reorder links or all L=0
            pipeline_state = np.array([])
            # Check if pipeline_obs_length was 0 in this case
            if self.pipeline_obs_length != 0:
                 print(f"Warning: pipeline_components is empty but pipeline_obs_length={self.pipeline_obs_length}")
                 # Create empty array of correct expected length if needed
                 pipeline_state = np.zeros(self.pipeline_obs_length)


        # Concatenate all parts
        # Ensure components are float32 before concatenating
        obs = np.concatenate([
            demand_backlog_component.astype(np.float32),
            inventory_component.astype(np.float32),
            pipeline_state.astype(np.float32)
        ])

        # Ensure final shape and dtype match the observation space
        if obs.shape[0] != self.obs_dim:
             # This indicates a mismatch between calculated obs_dim and actual state construction
             print(f"Warning: Observation dimension mismatch. Expected {self.obs_dim}, Got {obs.shape[0]}")
             # Attempt to pad or truncate (crude fix, indicates deeper issue)
             if obs.shape[0] < self.obs_dim:
                 padding = np.zeros(self.obs_dim - obs.shape[0], dtype=np.float32)
                 obs = np.concatenate([obs, padding])
             else:
                 obs = obs[:self.obs_dim]

        return obs.astype(self.observation_space.dtype) # Final cast

    def _get_info(self) -> Dict[str, Any]:
        """ Returns auxiliary information about the current state. """
        # Use .iloc for safety if index might become non-sequential later
        info = {
            "period": self.period,
            "inventory": self.X.iloc[self.period].to_dict(),
            "pipeline": self.Y.iloc[self.period].to_dict(),
            "backlog_start": self.U.iloc[self.period].to_dict()
        }
        # Add previous step results if available
        if self.period > 0:
            t_prev = self.period - 1
            info.update({
                "demand_prev": self.D.iloc[t_prev].to_dict(),
                "sales_prev": self.S.iloc[t_prev].to_dict(),
                "profit_node_prev": self.P.iloc[t_prev].to_dict(),
                "profit_total_prev": self.P.iloc[t_prev].sum()
            })
        return info


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """ Advances the environment by one time step. """
        t = self.period

        # --- 0) Place Orders ---
        # Convert action array to dictionary keyed by reorder links
        action_dict = {link: action[i] for i, link in enumerate(self.reorder_links)}

        # Process orders based on supplier type and constraints
        # Keep track of inventory used for production/sales from each node
        inventory_consumed = pd.Series(0.0, index=self.main_nodes)
        # Factory capacity is a per-period limit shared across all outgoing links.
        factory_output_committed = pd.Series(0.0, index=self.factory)

        for edge, request_raw in action_dict.items():
            request = max(0, round(request_raw)) # Ensure non-negative integer request
            supplier, purchaser = edge

            order_fulfilled = 0
            if supplier in self.rawmat:
                # Unlimited raw materials - order is always fulfilled
                order_fulfilled = request
            elif supplier in self.distrib or supplier in self.factory:
                # Limited by supplier's on-hand inventory at START of period t
                # We need to account for inventory potentially consumed by other orders from the same supplier in this loop
                inv_supplier_available = self.X.loc[t, supplier] - inventory_consumed[supplier]
                order_available_inv = max(0, inv_supplier_available) # Can't use negative inventory

                order_available = order_available_inv # Start with inventory constraint

                if supplier in self.factory:
                    # Also limited by capacity and yield
                    cap_supplier = self.graph.nodes[supplier]['C']
                    yield_supplier = self.graph.nodes[supplier]['v']
                    # Calculate max producible based on *remaining* inventory (after fulfilling previous orders)
                    max_producible_inv = yield_supplier * order_available_inv
                    # All outgoing orders compete for the same factory capacity.
                    max_producible_cap = max(0, cap_supplier - factory_output_committed[supplier])
                    max_producible = min(max_producible_cap, max_producible_inv)

                    order_available = min(order_available, max_producible) # Limited by production capability

                # Fulfill the minimum of requested and available
                order_fulfilled = min(request, order_available)

                # Update inventory consumed *for this supplier*
                node_yield = self.graph.nodes[supplier].get('v', 1.0) # Yield is 1 for distributors
                inventory_consumed[supplier] += order_fulfilled / node_yield
                if supplier in self.factory:
                    factory_output_committed[supplier] += order_fulfilled

            # Store the *fulfilled* order amount in R
            self.R.loc[t, edge] = order_fulfilled
            # Store the *fulfilled* amount as sales from the supplier in S
            self.S.loc[t, edge] = order_fulfilled

        # --- 1) Receive Deliveries & Update Pipeline Inventory ---
        # Update pipeline dataframe (Y) for start of t+1 *before* calculating arrivals
        for edge in self.reorder_links:
             supplier, purchaser = edge
             L = self.lead_times[edge]
             delivery_arriving_now = 0
             # Check if an order placed L periods ago exists and is arriving
             if t - L >= 0:
                 # Ensure index t-L is valid for R dataframe (index 0 to T-1)
                  if t - L < self.num_periods:
                     delivery_arriving_now = self.R.loc[t - L, edge]
             else: # t < L, no arrival possible yet
                 delivery_arriving_now = 0

             # Y[t+1] = Y[t] - arriving_now + newly_placed_order_R[t]
             # Ensure indices t and t+1 are valid for Y dataframe (index 0 to T)
             if t + 1 <= self.num_periods:
                 self.Y.loc[t + 1, edge] = self.Y.loc[t, edge] - delivery_arriving_now + self.R.loc[t, edge]
             else: # Avoid writing past end of simulation buffer
                 pass


        # --- Update On-Hand Inventory (X) for start of t+1 ---
        # X[t+1] = X[t] + (arrivals in t) - (consumption in t)
        arrivals_in_t = pd.Series(0.0, index=self.main_nodes)
        for node in self.main_nodes:
             for supplier in self.graph.predecessors(node):
                 edge = (supplier, node)
                 if edge in self.reorder_links:
                     L = self.lead_times[edge]
                     if t - L >= 0 and t - L < self.num_periods: # Check if arrival is due and valid index
                         arrivals_in_t[node] += self.R.loc[t - L, edge]

        # Consumption was calculated during order fulfillment (`inventory_consumed`)
        # Ensure index t+1 is valid for X dataframe (index 0 to T)
        if t + 1 <= self.num_periods:
             self.X.loc[t + 1] = self.X.loc[t] + arrivals_in_t - inventory_consumed
        else: # Avoid writing past end of simulation buffer
            pass

        # --- 2 & 3) Market Demand & Fulfillment ---
        # Use inventory available *after* arrivals but *before* market sales (X[t+1] currently holds this)
        X_before_market_sales = self.X.loc[t + 1].copy() if t + 1 <= self.num_periods else self.X.loc[t].copy() # Use copy

        for edge in self.retail_links:
            retailer, market = edge

            # Determine demand for this period using the stored sampler function
            demand_realized = max(0, int(round(self.graph.edges[edge]['_sample_demand']())))
            self.D.loc[t, edge] = demand_realized

            # Calculate total demand to fill (current + backlog from start of t, which is U[t])
            demand_to_fill = demand_realized + self.U.loc[t, edge]

            # Satisfy demand up to available inventory at the retailer
            inv_retailer = max(0, X_before_market_sales[retailer]) # Use positive inventory only
            sales_to_market = min(demand_to_fill, inv_retailer)

            # Update sales dataframe S for the market link
            self.S.loc[t, edge] = sales_to_market

            # Update the temporary inventory tracker after market sales
            X_before_market_sales[retailer] -= sales_to_market

            # --- 4) Calculate Unfulfilled Demand / Backlog ---
            unfulfilled = demand_to_fill - sales_to_market
             # Ensure index t+1 is valid for U dataframe (index 0 to T)
            if t + 1 <= self.num_periods:
                if self.backlog:
                    self.U.loc[t + 1, edge] = unfulfilled # Store backlog for next period
                else:
                    self.U.loc[t + 1, edge] = 0 # Lost sales, no backlog carries over
                    # LS dataframe could be added if explicit lost sales tracking is needed
            else: # Avoid writing past end of simulation buffer
                 pass


        # Final update to official inventory dataframe X[t+1] after demand fulfillment
        if t + 1 <= self.num_periods:
             self.X.loc[t+1] = X_before_market_sales
        else: # Avoid writing past end of simulation buffer
            pass


        # --- 5) Calculate Profit ---
        total_profit_period = 0
        for node in self.main_nodes:
             # Ensure indices t, t+1 are valid before accessing dataframes
            if t < self.num_periods and t + 1 <= self.num_periods:
                # Sales Revenue (from node selling to its successors)
                SR = sum(self.graph.edges[node, k]['p'] * self.S.loc[t, (node, k)]
                         for k in self.graph.successors(node) if (node, k) in self.network_links)

                # Purchasing Costs (from node buying from its predecessors via reorder links)
                PC = sum(self.graph.edges[k, node]['p'] * self.R.loc[t, (k, node)]
                         for k in self.graph.predecessors(node) if (k, node) in self.reorder_links)

                # Holding Costs (on-hand at end of t (i.e., X[t+1]) + pipeline at end of t (i.e., Y[t+1]))
                HC_on_hand = self.graph.nodes[node]['h'] * max(0, self.X.loc[t + 1, node]) # Cost on positive inventory
                HC_pipeline = sum(self.graph.edges[k, node]['g'] * max(0, self.Y.loc[t + 1, (k, node)]) # Cost on positive pipeline inventory
                                  for k in self.graph.predecessors(node) if (k, node) in self.reorder_links)
                HC = HC_on_hand + HC_pipeline

                # Operating Costs (for factories, based on amount produced/sold)
                OC = 0
                if node in self.factory:
                    node_yield = self.graph.nodes[node]['v']
                    total_sold_by_node = sum(self.S.loc[t, (node, k)] for k in self.graph.successors(node) if (node,k) in self.network_links)
                    # Cost applied per unit of *input* consumed. Input = total_sold / yield
                    OC = self.graph.nodes[node]['o'] * (total_sold_by_node / node_yield) if node_yield > 0 else 0

                # Unfulfilled Penalty (for retailers, based on unfulfilled demand U calculated for *this period*)
                UP = 0
                if node in self.retail:
                     # U[t+1] holds the unfulfilled amount *from period t*
                     # Penalty applies to the unfulfilled amount in the current period t
                     UP = sum(self.graph.edges[node, k]['b'] * self.U.loc[t + 1, (node, k)]
                              for k in self.graph.successors(node) if (node, k) in self.retail_links)

                node_profit = SR - PC - OC - HC - UP
                self.P.loc[t, node] = node_profit
                total_profit_period += node_profit
            else: # Should not happen if loop runs correctly, but prevents index error
                 pass


        # Apply discount factor
        discounted_total_profit = (self.alpha ** t) * total_profit_period

        # --- Prepare return values ---
        self.period += 1
        terminated = False # No natural end state
        truncated = self.period >= self.num_periods

        # Get next observation and info
        observation = self._get_obs()
        info = self._get_info()
        # Add step-specific info
        info['profit_period_undiscounted'] = total_profit_period
        info['profit_period_discounted'] = discounted_total_profit

        reward = float(discounted_total_profit) # Ensure reward is float

        return observation, reward, terminated, truncated, info


    def sample_action(self) -> np.ndarray:
        """ Samples a random action from the action space. """
        return self.action_space.sample()

    def render(self, mode="human"):
        """ Renders the environment (e.g., prints state or plots network). """
        if mode == "human":
            # Option 1: Print basic info for the *start* of the current period
            t_current = self.period # period has already been incremented for the *next* step
            t_print = min(t_current, self.num_periods) # Don't index past end of arrays

            print(f"--- START of Period: {t_print} ---")
            if t_print <= self.num_periods:
                 print("Inventory (X):")
                 print(self.X.loc[t_print])
                 print("\nPipeline (Y):")
                 print(self.Y.loc[t_print])
                 print("\nBacklog (U):")
                 print(self.U.loc[t_print])

            if t_print > 0:
                t_prev = t_print - 1
                print(f"\n--- Results from Previous Period ({t_prev}) ---")
                print(f"  Demand (D): {self.D.loc[t_prev].to_dict()}")
                print(f"  Sales (S): {self.S.loc[t_prev].to_dict()}")
                print(f"  Profit (P): {self.P.loc[t_prev].sum():.2f}")

            # Option 2: Plot network (can be slow if called every step)
            # self.plot_network()
        else:
            # Handle other modes if needed, or rely on superclass render
             return super().render(mode=mode)


    def plot_network(self):
        """ Plots the network structure using matplotlib. """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Cannot plot network.")
            return

        plt.figure(figsize=(12, 8))

        # Assign layers (simple heuristic based on type)
        node_layers = {}
        for node in self.graph.nodes():
            if node in self.rawmat: node_layers[node] = 0
            elif node in self.factory: node_layers[node] = 1
            # Separate distributors based on whether they are retailers
            elif node in self.retail: node_layers[node] = 3
            elif node in self.distrib: node_layers[node] = 2 # Non-retail distributors
            elif node in self.market: node_layers[node] = 4
            else: node_layers[node] = 2 # Default layer if unclassified (shouldn't happen with current logic)
        nx.set_node_attributes(self.graph, node_layers, "layer")

        # Use the 'layer' attribute for layout
        pos = nx.multipartite_layout(self.graph, subset_key='layer')

        # Draw nodes with different colors based on type
        node_colors = []
        node_labels = {}
        for node in self.graph.nodes():
            node_labels[node] = f"{node}" # Basic label is node number
            if node in self.rawmat:
                 color = 'gray'
                 node_labels[node] += "\n(RawM)"
            elif node in self.factory:
                 color = 'skyblue'
                 node_labels[node] += "\n(Fact)"
            elif node in self.retail:
                 color = 'lightgreen'
                 node_labels[node] += "\n(Retail)"
            elif node in self.distrib:
                 color = 'khaki' # Non-retail distributors
                 node_labels[node] += "\n(Dist)"
            elif node in self.market:
                 color = 'salmon'
                 node_labels[node] += "\n(Market)"
            else: color = 'pink' # Unclassified?
            node_colors.append(color)

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=1500, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray', node_size=1500)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=9)

        # Add edge labels (optional, can get cluttered - maybe just lead time?)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
             label = ""
             if 'L' in data: label += f"L={data['L']}"
             #if 'p' in data: label += f"\np={data['p']:.2f}" # Price adds clutter
             if label: edge_labels[(u, v)] = label

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

        plt.title("Supply Network Structure")
        # Improve axis labels/titles maybe? multipartite doesn't use axes directly
        plt.text(0.5, 1.01, "Upstream (Raw Materials) -> Downstream (Market)", ha='center', transform=plt.gca().transAxes)
        plt.box(False) # Remove frame
        plt.show()


    def close(self):
        """ Perform any necessary cleanup. """
        pass

# --- Subclasses for Backlog vs Lost Sales ---

class NetInvMgmtBacklogEnv(NetInvMgmtMasterEnv):
    """ Network inventory management environment with backlogging enabled. """
    def __init__(self, *args, **kwargs):
        # Force the requested inventory policy at construction time.
        kwargs['backlog'] = True
        super().__init__(*args, **kwargs)

class NetInvMgmtLostSalesEnv(NetInvMgmtMasterEnv):
    """ Network inventory management environment with lost sales. """
    def __init__(self, *args, **kwargs):
        # Preserve lost-sales semantics, regardless of supplied kwargs/config.
        kwargs['backlog'] = False
        super().__init__(*args, **kwargs)

        # Adjust observation space lower bound: Backlog component should be non-negative
        obs_low_ls = self.observation_space.low.copy()
        obs_low_ls[0:len(self.retail_links)] = 0.0
        self.observation_space = spaces.Box(
            low=obs_low_ls,
            high=self.observation_space.high,
            shape=(self.obs_dim,),
            dtype=self.observation_space.dtype)


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing Gymnasium-compatible NetInvMgmt Environments...")

    # Test Backlog Env using default graph
    print("\n--- Testing Backlog Environment (Default Graph) ---")
    env_backlog = NetInvMgmtBacklogEnv(num_periods=15) # Shorter simulation

    obs, info = env_backlog.reset(seed=42)
    print(f"Observation Space: {env_backlog.observation_space}")
    print(f"Action Space: {env_backlog.action_space}")
    print(f"Initial Obs Dim: {obs.shape}")
    # print(f"Initial Obs (Backlog): {obs}")
    # print(f"Initial Info (Backlog): {info}")

    total_reward_backlog = 0
    terminated = False
    truncated = False

    for i in range(env_backlog.num_periods):
        action = env_backlog.sample_action() # Sample random action
        try: # Add try-except block around step for easier debugging
            obs, reward, terminated, truncated, info = env_backlog.step(action)
            total_reward_backlog += reward
            if (i+1) % 5 == 0: # Print occasionally
                 print(f"  Step {i+1}, Reward: {reward:.2f}, Total Reward: {total_reward_backlog:.2f}")
                 # env_backlog.render() # Optionally render state printout
            if terminated or truncated:
                break
        except Exception as e:
             print(f"Error during step {i+1}: {e}")
             # Optionally print more debugging info like current state/action
             print(f"  Action taken: {action}")
             print(f"  Current Info: {info}") # Info from *before* the failed step
             import traceback
             traceback.print_exc()
             break # Stop simulation on error

    print(f"\nEpisode Finished (Backlog). Total Reward: {total_reward_backlog:.2f}")
    # print(f"Final Info (Backlog): {info}")
    env_backlog.close()

    # Optionally plot the network structure
    try:
        env_backlog.plot_network()
    except Exception as e:
        print(f"Error during plot_network for Backlog Env: {e}")


    # Test Lost Sales Env
    print("\n--- Testing Lost Sales Environment (Default Graph) ---")
    env_lost_sales = NetInvMgmtLostSalesEnv(num_periods=15)

    obs, info = env_lost_sales.reset(seed=123)
    print(f"Initial Obs Dim: {obs.shape}")

    total_reward_lost_sales = 0
    terminated = False
    truncated = False

    for i in range(env_lost_sales.num_periods):
        action = env_lost_sales.sample_action()
        try:
            obs, reward, terminated, truncated, info = env_lost_sales.step(action)
            total_reward_lost_sales += reward
            if (i+1) % 5 == 0:
                 print(f"  Step {i+1}, Reward: {reward:.2f}, Total Reward: {total_reward_lost_sales:.2f}")
            if terminated or truncated:
                break
        except Exception as e:
             print(f"Error during step {i+1}: {e}")
             print(f"  Action taken: {action}")
             print(f"  Current Info: {info}")
             import traceback
             traceback.print_exc()
             break

    print(f"\nEpisode Finished (Lost Sales). Total Reward: {total_reward_lost_sales:.2f}")
    env_lost_sales.close()

    print("\nTesting complete.")
