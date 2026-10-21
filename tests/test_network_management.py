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


    def test_factory_capacity_shared_across_outgoing_links(self):
        for env_type in (NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(num_periods=2)
                # Factory 4 supplies both distributor 2 and distributor 3.
                env.graph.nodes[4]["C"] = 5
                env.reset(seed=42)

                orders = np.zeros(env.action_space.shape, dtype=np.float32)
                orders[env.reorder_links.index((4, 2))] = 4
                orders[env.reorder_links.index((4, 3))] = 4
                env.step(orders)

                self.assertEqual(env.R.loc[0, (4, 2)], 4)
                self.assertEqual(env.R.loc[0, (4, 3)], 1)
                self.assertEqual(env.R.loc[0, (4, 2)] + env.R.loc[0, (4, 3)], 5)
                self.assertEqual(env.X.loc[1, 4], 395)

                # Capacity is available again in a fresh production period.
                env.step(orders)
                self.assertEqual(env.R.loc[1, (4, 2)] + env.R.loc[1, (4, 3)], 5)
                self.assertEqual(env.X.loc[2, 4], 390)

    def test_factory_capacity_is_independent_by_factory(self):
        env = NetInvMgmtBacklogEnv(num_periods=1)
        env.graph.nodes[4]["C"] = 5
        env.graph.nodes[5]["C"] = 3
        env.reset(seed=42)
        orders = np.zeros(env.action_space.shape, dtype=np.float32)
        for edge in ((4, 2), (4, 3), (5, 2)):
            orders[env.reorder_links.index(edge)] = 4
        env.step(orders)

        self.assertEqual(env.R.loc[0, (4, 2)] + env.R.loc[0, (4, 3)], 5)
        self.assertEqual(env.R.loc[0, (5, 2)], 3)
        self.assertEqual(env.X.loc[1, 4], 395)
        self.assertEqual(env.X.loc[1, 5], 347)


    def test_lost_sales_incur_penalty_without_backlog(self):
        def run(penalty):
            env = NetInvMgmtLostSalesEnv(num_periods=2)
            env.reset(seed=42)
            env.graph.edges[(1, 0)]["_sample_demand"] = lambda: 200
            env.graph.edges[(1, 0)]["b"] = penalty
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            first_reward = env.step(action)[1]
            first_backlog = env.U.loc[1, (1, 0)]
            second_reward = env.step(action)[1]
            second_backlog = env.U.loc[2, (1, 0)]
            return first_reward, second_reward, first_backlog, second_backlog

        without_penalty = run(0.0)
        with_penalty = run(0.5)

        self.assertEqual(with_penalty[2:], (0, 0))
        # Period 0: 200 demand - 100 stock = 100 units lost.
        self.assertAlmostEqual(without_penalty[0] - with_penalty[0], 50.0)
        # Period 1: zero stock, so all 200 units are lost (not backlogged).
        self.assertAlmostEqual(without_penalty[1] - with_penalty[1], 100.0)

    def test_backlog_penalty_remains_based_on_unfilled_demand(self):
        def run(penalty):
            env = NetInvMgmtBacklogEnv(num_periods=2)
            env.reset(seed=42)
            env.graph.edges[(1, 0)]["_sample_demand"] = lambda: 200
            env.graph.edges[(1, 0)]["b"] = penalty
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            first_reward = env.step(action)[1]
            first_backlog = env.U.loc[1, (1, 0)]
            second_reward = env.step(action)[1]
            second_backlog = env.U.loc[2, (1, 0)]
            return first_reward, second_reward, first_backlog, second_backlog

        without_penalty = run(0.0)
        with_penalty = run(0.5)

        self.assertEqual(with_penalty[2:], (100, 300))
        self.assertAlmostEqual(without_penalty[0] - with_penalty[0], 50.0)
        self.assertAlmostEqual(without_penalty[1] - with_penalty[1], 150.0)


if __name__ == "__main__":
    unittest.main()
