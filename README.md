# Gymnasium-Compatible Inventory Management Environments & Benchmarks

This repository provides implementations of classic inventory management environments, adapted from the original [OR-Gym library](https://github.com/hubbs5/or-gym), updated for compatibility with the [Gymnasium](https://gymnasium.farama.org/) API (the successor to OpenAI Gym). It also includes comprehensive benchmarking scripts to compare various heuristic, optimization-inspired, and Reinforcement Learning (RL) policies on these environments.

The original environments are based on the work by Hubbs et al. (2020):
- Paper: https://arxiv.org/abs/2008.04001
- Original OR-Gym GitHub (using OpenAI Gym): https://github.com/hubbs5/or-gym

**Environments Adapted:**
1.  **Newsvendor (`newsvendor.py`):** Multi-period newsvendor problem with lead times and stochastic Poisson demand (based on Balaji et al. 2019, https://arxiv.org/abs/1911.10641).
2.  **Inventory Management (`inventory_management.py`):** Multi-period, multi-echelon inventory system for a single product. Includes `InvManagementBacklogEnv` and `InvManagementLostSalesEnv`.
3.  **Network Inventory Management (`network_management.py`):** Multi-period, multi-node inventory system with a network structure (factories, distributors, retailers, markets). Includes `NetInvMgmtBacklogEnv` and `NetInvMgmtLostSalesEnv`.

## Features

*   **Gymnasium Compatible:** Environments adhere to the modern Gymnasium API standard (`reset` returns `obs, info`, `step` returns `obs, reward, terminated, truncated, info`).
*   **Three Core Environments:** Covers single-item, multi-echelon, and network inventory problems.
*   **Backlog & Lost Sales Variants:** Specific environment classes (`*BacklogEnv`, `*LostSalesEnv`) implement these dynamics.
*   **Comprehensive Benchmarking:** Includes dedicated scripts (`benchmark_*.py`) for each environment variant, comparing various agents:
    *   **Baselines:** Random Agent.
    *   **Heuristics:** Relevant heuristics adapted for each environment type (e.g., Order-Up-To, Classic Newsvendor, (s,S) for Newsvendor; Base Stock, Constant Order for multi-echelon/network).
    *   **Stable Baselines3 Agents:** PPO, SAC, TD3, A2C, DDPG (plus example variations like LSTM). Scripts suffixed with `_sb3_rllib.py` explicitly include these. Simpler benchmark scripts might contain a subset.
    *   **Ray RLlib Agents:** PPO, SAC (plus framework to add more). Scripts suffixed with `_sb3_rllib.py` explicitly include these.
*   **Detailed Reporting:** Benchmarks generate:
    *   Summary tables comparing agents on average reward, consistency (std dev, min/max), operational metrics (service level, stockouts, inventory), and time (training/evaluation). Saved to CSV.
    *   Raw results per evaluation episode (CSV).
    *   Detailed step-by-step data (optional, JSON Lines).
    *   Comparison plots (Reward distribution boxplots, Reward vs. Operational Metrics scatter plots, Timing bar charts, RL learning curves). Saved to PNG.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/r2barati/or-gym-inventory.git
    cd or-gym-inventory
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Ensure you have Python 3.8+ installed.
    *   Upgrade pip: `python -m pip install --upgrade pip`
    *   Install core requirements and RL libraries:
        ```bash
        # Install base requirements (Gymnasium, Numpy, Scipy, Pandas, NetworkX, Matplotlib, Seaborn)
        pip install -r requirements.txt

        # --- Stable-Baselines3 uses PyTorch, not TensorFlow ---
        pip install 'stable-baselines3[extra]'

        # --- Install Ray RLlib (if running *_sb3_rllib.py scripts) ---
        # Choose ONE framework (can match SB3 or be different, installing both is possible)
        pip install "ray[rllib]" torch torchvision torchaudio
        # OR
        # pip install "ray[rllib]" tensorflow
        ```
        *(Note: `requirements.txt` should contain `gymnasium`, `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `seaborn`)*

## Usage

### 1. Using the Environments Directly

You can import and use the specific environment classes like any standard Gymnasium environment:

```python
import gymnasium as gym
# Make sure the relevant python file (e.g., inventory_management.py) is in your path
from inventory_management import InvManagementLostSalesEnv # Example

# Configuration dictionary (optional, overrides defaults)
env_config = {
    'periods': 50,
    'I0': [50, 50],
    'L': [2, 4],
    'c': [30, 30]
    # ... other parameters specific to the environment ...
}

# Create the environment
# env = InvManagementLostSalesEnv(env_config=env_config)
env = InvManagementLostSalesEnv() # Use defaults

# Standard Gymnasium loop
observation, info = env.reset(seed=42)
# ... (rest of loop as shown in previous README version) ...
env.close()
```

### 2. Running the Benchmarks

The repository includes dedicated benchmark scripts for each environment variant. The scripts ending in `_sb3_rllib.py` contain the most comprehensive set of agents, including both Stable Baselines3 and Ray RLlib implementations. Simpler scripts might focus only on SB3 or heuristics.

**Choose the script corresponding to the environment you want to benchmark:**

*   **Newsvendor (Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_newsvendor_sb3_rllib.py
    ```
*   **Inventory Management (Backlog, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_InvManagementBacklogEnv_sb3_rllib.py
    ```
*   **Inventory Management (Lost Sales, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_InvManagementLostSalesEnv_sb3_rllib.py
    ```
*   **Network Inventory Management (Backlog, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_NetInvMgmtBacklogEnv_sb3_rllib.py
    ```
*   **Network Inventory Management (Lost Sales, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_NetInvMgmtLostSalesEnv_sb3_rllib.py
    ```

**Note:** Running the benchmarks, especially those involving RL agent training (`RL_TRAINING_TIMESTEPS` > 0), can take a significant amount of time, potentially hours depending on the number of steps, agents, and your hardware. Start with lower `RL_TRAINING_TIMESTEPS` (e.g., 10000-50000) in the script config to test functionality.

## Environments Overview

*   **`NewsvendorEnv` (`newsvendor.py`):**
    *   Single product, single location.
    *   Agent decides order quantity each period.
    *   Stochastic demand (Poisson).
    *   Fixed lead time for orders.
    *   Costs: Purchase, Holding, Stockout (Lost Sales Penalty).
    *   Observation includes costs, demand mean, and pipeline inventory.
    *   Action is a single continuous value (order quantity).
*   **`InvManagement...Env` (`inventory_management.py`):**
    *   Single product, multi-echelon linear supply chain (e.g., Retailer -> Distributor -> Manufacturer).
    *   Agent decides order quantity for each stage (except the raw material source) each period.
    *   Stochastic demand (configurable distribution) only at the retailer (stage 0).
    *   Lead times between stages.
    *   Production capacities at manufacturing stages.
    *   Costs: Purchase/Replenishment, Holding (on-hand), Backlog/Lost Sales.
    *   Observation includes on-hand inventory and recent order history (pipeline).
    *   Action is a vector of continuous/integer values (order quantities per stage).
    *   Variants: `InvManagementBacklogEnv` and `InvManagementLostSalesEnv`.
*   **`NetInvMgmt...Env` (`network_management.py`):**
    *   Single product, arbitrary network structure (defined via `networkx` graph).
    *   Nodes can be Raw Material, Factory, Distributor, Retailer, Market.
    *   Agent decides order quantity for each valid *link* between supplying/receiving nodes.
    *   Stochastic demand (configurable) occurs at links between Retailers and Markets.
    *   Lead times associated with links.
    *   Production capacities and yields at factory nodes.
    *   Costs: Purchase/Replenishment (link), Operating (factory), Holding (on-hand at node, pipeline on link), Backlog/Lost Sales (market link).
    *   Observation includes market backlog/demand, node inventories, and pipeline history per link.
    *   Action is a vector of continuous values (order quantity per link).
    *   Variants: `NetInvMgmtBacklogEnv` and `NetInvMgmtLostSalesEnv`.

*(Refer to the docstrings within each environment file (`.py`) for details on observation/action spaces, reward calculation, and specific parameters.)*

## Benchmarking Details

The `benchmark_*.py` scripts provide a framework to compare agents. The `_sb3_rllib.py` versions are the most comprehensive.

**Agents Included:**

*   **Heuristics:**
    *   `RandomAgent`: Random actions.
    *   `ConstantOrderAgent`: Orders a fixed fraction (Network env).
    *   `OrderUpToHeuristicAgent`: Targets expected demand over L+1 (Newsvendor).
    *   `ClassicNewsvendorAgent`: Uses critical ratio and demand quantile (Newsvendor).
    *   `sSPolicyAgent`: Orders up to S if below s (Newsvendor).
    *   `BaseStockAgent`: Simple independent base stock per stage (InvManagement).
*   **Stable Baselines3 (SB3):**
    *   PPO, SAC, TD3, A2C, DDPG
    *   Example variations (LSTM policy, different buffer/LR/network sizes).
*   **Ray RLlib:**
    *   PPO, SAC (Examples)
    *   Framework allows easy addition of others (TD3, DDPG, APEX, IMPALA etc.).

**Metrics Collected (per Agent):**

*   Average Total Reward (and Median, Std Dev, Min, Max)
*   Average Service Level (Fill Rate, usually at retailer/market)
*   Average/Total Stockout Quantity
*   Average Ending Inventory
*   Average Evaluation Time per Episode
*   Total Training Time (for RL agents)
*   Evaluation Success Rate

## Results Interpretation

The benchmark scripts save results into subdirectories named like `benchmark_<ENV_NAME>_combined/results/`:

*   **`*_summary.csv`:** A table summarizing the average performance and time metrics for each agent, sorted by average reward. This is the main comparison table.
*   **`*_raw_summary.csv`:** Contains the results (total reward, metrics) for *each individual evaluation episode* for every agent. Useful for statistical analysis or plotting distributions.
*   **`*_step_details.jsonl`:** (Optional, if `COLLECT_STEP_DETAILS=True`) Contains detailed data for *every step* within every evaluation episode (reward, action, demand, sales, etc.). Can be very large but useful for deep dives.
*   **`*_rewards_boxplot.png`:** Visualizes the distribution of total rewards achieved by each agent across the evaluation episodes. Helps assess consistency.
*   **`*_reward_vs_service.png` / `*_reward_vs_inventory.png`:** Scatter plots showing trade-offs between average reward and key operational metrics.
*   **`*_eval_time_log.png` / `*_train_time.png`:** Bar charts comparing evaluation and training times.
*   **`*_learning_curves.png`:** Shows the training progress (reward vs. timesteps) for the RL agents, plotted from SB3 Monitor files and/or custom RLlib logs.

## Dependencies

*   Python 3.8+
*   gymnasium
*   numpy
*   scipy
*   pandas
*   networkx (for NetInvMgmt environment)
*   matplotlib
*   seaborn
*   stable-baselines3[extra] (`pip install stable-baselines3[extra] torch # or tensorflow`)
*   ray[rllib] (`pip install "ray[rllib]" torch # or tensorflow`)

See `requirements.txt` for base dependencies.

## References

*   Hubbs, C., Perez, H. D., Sarwar, O., Li, C., & Papageorgiou, D. (2020). OR-Gym: A Reinforcement Learning Library for Operations Research Problems. *arXiv preprint arXiv:2008.04001*. ([Link](https://arxiv.org/abs/2008.04001))

## License

MIT License
