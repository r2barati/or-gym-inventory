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


    def test_explicit_zero_demand_is_not_randomized(self):
        for env_type in (NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(num_periods=2, user_D={(1, 0): [0, 0]})
                env.reset(seed=42)
                no_orders = np.zeros(env.action_space.shape, dtype=np.float32)
                env.step(no_orders)
                env.step(no_orders)
                self.assertEqual(env.D.loc[0, (1, 0)], 0)
                self.assertEqual(env.D.loc[1, (1, 0)], 0)

    def test_env_config_explicit_demand_including_zero(self):
        config = {"num_periods": 2, "user_D": {(1, 0): [0, 3]}}
        env = NetInvMgmtBacklogEnv(env_config=config)
        env.reset(seed=42)
        no_orders = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(no_orders)
        env.step(no_orders)
        self.assertEqual(list(env.D[(1, 0)]), [0, 3])

    def test_unset_demand_still_uses_sampler(self):
        env = NetInvMgmtBacklogEnv(num_periods=2)
        # The default graph has a placeholder all-zero user_D, not a supplied path.
        env.graph.edges[(1, 0)]["demand_dist_func"] = lambda **kwargs: 7
        env.reset(seed=42)
        no_orders = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(no_orders)
        self.assertEqual(env.D.loc[0, (1, 0)], 7)

    def test_zero_demand_path_length_must_match_horizon(self):
        with self.assertRaisesRegex(AssertionError, "user_D length"):
            NetInvMgmtBacklogEnv(num_periods=2, user_D={(1, 0): [0]})


    def test_network_raw_material_orders_respect_action_bounds(self):
        for env_type in (NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(num_periods=1)
                env.reset(seed=42)
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                idx = env.reorder_links.index((7, 4))
                action[idx] = 1_000_000.0
                expected = float(round(env.action_space.high[idx]))
                observation, _, _, truncated, _ = env.step(action)

                self.assertTrue(truncated)
                self.assertEqual(env.R.loc[0, (7, 4)], expected)
                self.assertEqual(env.S.loc[0, (7, 4)], expected)
                self.assertEqual(env.X.loc[1, 4], 400 + expected)
                self.assertTrue(env.observation_space.contains(observation))

    def test_network_action_vector_rejects_incorrect_shape(self):
        for env_type in (NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv):
            for extra in (-1, 1):
                with self.subTest(env_type=env_type.__name__, extra=extra):
                    env = env_type(num_periods=1)
                    env.reset(seed=42)
                    bad_action = np.zeros(len(env.reorder_links) + extra)
                    with self.assertRaisesRegex(ValueError, "Expected action shape"):
                        env.step(bad_action)
                    self.assertEqual(env.period, 0)
                    self.assertEqual(env.R.to_numpy().sum(), 0)

    def test_network_nonfinite_orders_rejected_before_state_changes(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                env = NetInvMgmtBacklogEnv(num_periods=1)
                env.reset(seed=42)
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                action[0] = value
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    env.step(action)
                self.assertEqual(env.period, 0)
                self.assertEqual(env.R.to_numpy().sum(), 0)

    def test_negative_raw_material_orders_are_clipped_to_zero(self):
        env = NetInvMgmtLostSalesEnv(num_periods=1)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[env.reorder_links.index((7, 4))] = -10
        env.step(action)
        self.assertEqual(env.R.loc[0, (7, 4)], 0)


    def test_fractional_factory_yield_ships_only_whole_units(self):
        for env_type in (NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(num_periods=2)
                env.graph.nodes[4]["I0"] = 5
                env.graph.nodes[4]["v"] = 0.5
                env.graph.nodes[4]["C"] = 50
                env.reset(seed=42)

                orders = np.zeros(env.action_space.shape, dtype=np.float32)
                orders[env.reorder_links.index((4, 2))] = 6
                observation, _, _, _, _ = env.step(orders)

                self.assertEqual(env.R.loc[0, (4, 2)], 2)
                self.assertEqual(env.S.loc[0, (4, 2)], 2)
                self.assertEqual(env.Y.loc[1, (4, 2)], 2)
                self.assertEqual(env.X.loc[1, 4], 1)
                self.assertTrue(env.observation_space.contains(observation))

    def test_shared_factory_input_inventory_is_not_overconsumed(self):
        for env_type in (NetInvMgmtBacklogEnv, NetInvMgmtLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(num_periods=2)
                env.graph.nodes[4]["I0"] = 15
                env.graph.nodes[4]["v"] = 0.5
                env.graph.nodes[4]["C"] = 50
                env.reset(seed=42)

                orders = np.zeros(env.action_space.shape, dtype=np.float32)
                orders[env.reorder_links.index((4, 2))] = 3
                orders[env.reorder_links.index((4, 3))] = 10
                env.step(orders)
                self.assertEqual(env.R.loc[0, (4, 2)], 3)
                self.assertEqual(env.R.loc[0, (4, 3)], 4)
                self.assertEqual(env.X.loc[1, 4], 1)
                self.assertGreaterEqual(env.X.loc[1, 4], 0)

                # Insufficient remaining input stock to produce a whole unit.
                env.step(orders)
                self.assertEqual(env.R.loc[1, (4, 2)], 0)
                self.assertEqual(env.R.loc[1, (4, 3)], 0)
                self.assertEqual(env.X.loc[2, 4], 1)


    def test_large_valid_demand_backlog_stays_within_observation_space(self):
        env = NetInvMgmtBacklogEnv(
            num_periods=2, user_D={(1, 0): [1_000_000, 2_000_000]}
        )
        self.assertTrue(np.isposinf(env.observation_space.high[0]))
        # Inventory and pipeline still have finite capacity-based bounds.
        self.assertTrue(np.all(np.isfinite(env.observation_space.high[1:])))

        initial_obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(initial_obs))
        no_orders = np.zeros(env.action_space.shape, dtype=np.float32)

        first_obs, first_reward, _, first_truncated, _ = env.step(no_orders)
        self.assertFalse(first_truncated)
        self.assertTrue(np.isfinite(first_reward))
        self.assertEqual(first_obs[0], 999_900)
        self.assertTrue(env.observation_space.contains(first_obs))

        second_obs, second_reward, _, second_truncated, _ = env.step(no_orders)
        self.assertTrue(second_truncated)
        self.assertTrue(np.isfinite(second_reward))
        self.assertEqual(second_obs[0], 2_999_900)
        self.assertTrue(env.observation_space.contains(second_obs))

    def test_large_valid_demand_in_lost_sales_mode_has_zero_backlog(self):
        env = NetInvMgmtLostSalesEnv(
            num_periods=2, user_D={(1, 0): [1_000_000, 2_000_000]}
        )
        self.assertTrue(np.isfinite(env.observation_space.high[0]))
        env.reset(seed=42)
        no_orders = np.zeros(env.action_space.shape, dtype=np.float32)

        for period in (1, 2):
            with self.subTest(period=period):
                obs, reward, _, _, _ = env.step(no_orders)
                self.assertEqual(obs[0], 0)
                self.assertEqual(env.U.loc[period, (1, 0)], 0)
                self.assertTrue(np.isfinite(reward))
                self.assertTrue(env.observation_space.contains(obs))


if __name__ == "__main__":
    unittest.main()
