"""Network inventory policies: backlogging must differ from lost sales."""

import unittest

import numpy as np

from network_management import NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv


class NetworkInventoryPolicyTests(unittest.TestCase):
    def test_lost_sales_never_carries_unfulfilled_demand(self):
        config = {"num_periods": 2, "backlog": True}
        env = NetInvMgmtLostSalesEnv(env_config=config)
        # Instantiating an environment must not change its caller's dictionary.
        self.assertEqual(config, {"num_periods": 2, "backlog": True})
        self.assertFalse(env.backlog)
        self.assertTrue(np.all(env.observation_space.low >= 0))

        env.reset(seed=42)
        # Deterministic demand deliberately exceeds available retailer stock.
        env.graph.edges[(1, 0)]["_sample_demand"] = lambda: 200
        zero_orders = np.zeros(env.action_space.shape, dtype=np.float32)
        first_observation, _, _, _, _ = env.step(zero_orders)
        self.assertEqual(env.U.loc[1, (1, 0)], 0)
        self.assertEqual(first_observation[0], 0)

        second_observation, _, _, truncated, _ = env.step(zero_orders)
        self.assertTrue(truncated)
        self.assertEqual(env.U.loc[2, (1, 0)], 0)
        self.assertEqual(second_observation[0], 0)

    def test_backlog_mode_carries_unfulfilled_demand(self):
        config = {"num_periods": 2, "backlog": False}
        env = NetInvMgmtBacklogEnv(env_config=config)
        self.assertEqual(config, {"num_periods": 2, "backlog": False})
        self.assertTrue(env.backlog)

        env.reset(seed=42)
        env.graph.edges[(1, 0)]["_sample_demand"] = lambda: 200
        zero_orders = np.zeros(env.action_space.shape, dtype=np.float32)

        first_observation, _, _, _, _ = env.step(zero_orders)
        self.assertEqual(env.U.loc[1, (1, 0)], 100)
        self.assertEqual(first_observation[0], 100)

        second_observation, _, _, truncated, _ = env.step(zero_orders)
        self.assertTrue(truncated)
        self.assertEqual(env.U.loc[2, (1, 0)], 300)
        self.assertEqual(second_observation[0], 300)


if __name__ == "__main__":
    unittest.main()
