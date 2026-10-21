"""Inventory conservation regressions for multi-echelon replenishment."""
import unittest

import numpy as np

from inventory_management import InvManagementBacklogEnv, InvManagementLostSalesEnv


class InventoryShipmentTests(unittest.TestCase):
    def make_env(self, env_type, initial_stock):
        n = len(initial_stock)
        env = env_type(env_config={
            "periods": 2,
            "I0": initial_stock,
            "r": [1.0] * (n + 1),
            "k": [0.0] * (n + 1),
            "h": [0.0] * n,
            "c": [50] * n,
            "L": [1] * n,
            "dist": 5,
            "user_D": [0, 0],
            "alpha": 1.0,
        })
        env.reset(seed=42)
        return env

    def test_supplier_stock_is_debited_for_downstream_order(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 10])
                _, _, _, _, info = env.step(np.array([4, 0], dtype=np.int64))
                np.testing.assert_array_equal(info["ending_inventory"], [0, 6])
                np.testing.assert_array_equal(env.R[0], [4, 0])

                # The four shipped units arrive at the retailer one period later.
                _, _, _, _, info = env.step(np.array([0, 0], dtype=np.int64))
                np.testing.assert_array_equal(info["ending_inventory"], [4, 6])

    def test_each_supplier_loses_only_its_own_shipments(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 10, 20])
                _, _, _, _, info = env.step(np.array([4, 5, 0], dtype=np.int64))
                np.testing.assert_array_equal(env.R[0], [4, 5, 0])
                np.testing.assert_array_equal(info["ending_inventory"], [0, 6, 15])
                self.assertEqual(sum(info["ending_inventory"]), 21)

                _, _, _, _, info = env.step(np.array([0, 0, 0], dtype=np.int64))
                np.testing.assert_array_equal(info["ending_inventory"], [4, 11, 15])
                self.assertEqual(sum(info["ending_inventory"]), 30)

    def test_supplier_cannot_ship_more_than_available_stock(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 2, 7])
                _, _, _, _, info = env.step(np.array([4, 5, 0], dtype=np.int64))
                np.testing.assert_array_equal(env.R[0], [2, 5, 0])
                np.testing.assert_array_equal(info["ending_inventory"], [0, 0, 2])
                self.assertTrue(np.all(info["ending_inventory"] >= 0))
                self.assertEqual(info["unfulfilled"][1], 2)


    def test_observation_history_is_newest_first_from_episode_start(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(env_config={
                    "periods": 4, "I0": [0, 20],
                    "r": [1.0, 1.0, 1.0], "k": [0.0] * 3,
                    "h": [0.0, 0.0], "c": [50, 50],
                    "L": [3, 3], "dist": 5, "user_D": [0] * 4,
                })
                initial, _ = env.reset(seed=42)
                np.testing.assert_array_equal(initial[2:], [0] * 6)
                for action, expected in [
                    ([1, 0], [1, 0, 0, 0, 0, 0]),
                    ([2, 0], [2, 0, 1, 0, 0, 0]),
                    ([3, 0], [3, 0, 2, 0, 1, 0]),
                    ([4, 0], [4, 0, 3, 0, 2, 0]),
                ]:
                    obs, _, _, _, _ = env.step(np.array(action, dtype=np.int64))
                    np.testing.assert_array_equal(obs[2:], expected)
                    self.assertTrue(env.observation_space.contains(obs))

    def test_observation_history_preserves_stage_order(self):
        env = InvManagementLostSalesEnv(env_config={
            "periods": 2, "I0": [0, 20, 30],
            "r": [1.0] * 4, "k": [0.0] * 4,
            "h": [0.0] * 3, "c": [50] * 3,
            "L": [2, 1, 2], "dist": 5, "user_D": [0, 0],
        })
        env.reset(seed=42)
        first_obs, _, _, _, _ = env.step(np.array([1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(first_obs[3:], [1, 2, 3, 0, 0, 0])
        second_obs, _, _, _, _ = env.step(np.array([4, 5, 6], dtype=np.int64))
        np.testing.assert_array_equal(second_obs[3:], [4, 5, 6, 1, 2, 3])

    def test_zero_lead_time_has_no_action_history_slots(self):
        env = InvManagementBacklogEnv(env_config={
            "periods": 2, "I0": [0, 20],
            "r": [1.0] * 3, "k": [0.0] * 3,
            "h": [0.0] * 2, "c": [50] * 2,
            "L": [0, 0], "dist": 5, "user_D": [0, 0],
        })
        initial, _ = env.reset(seed=42)
        self.assertEqual(initial.shape, (2,))
        new_obs, _, _, _, _ = env.step(np.array([1, 0], dtype=np.int64))
        self.assertEqual(new_obs.shape, (2,))
        self.assertTrue(env.observation_space.contains(new_obs))


    def test_actions_are_bounded_before_backlog_and_history(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = env_type(env_config={
                    "periods": 2, "I0": [0, 20],
                    "r": [1.0] * 3, "k": [0.0] * 3,
                    "h": [0.0] * 2, "c": [5, 7],
                    "L": [1, 1], "dist": 5, "user_D": [0, 0],
                })
                env.reset(seed=42)
                observation, _, _, _, info = env.step(
                    np.array([1_000_000, -3], dtype=np.int64)
                )
                np.testing.assert_array_equal(env.action_log[0], [5, 0])
                np.testing.assert_array_equal(env.R[0], [5, 0])
                self.assertEqual(info["unfulfilled"][1], 0)
                self.assertTrue(env.observation_space.contains(observation))
                np.testing.assert_array_equal(observation[2:], [5, 0])

    def test_invalid_action_shape_does_not_advance_episode(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 10])
                with self.assertRaisesRegex(ValueError, "Expected action shape"):
                    env.step(np.array([3], dtype=np.int64))
                self.assertEqual(env.period, 0)
                self.assertEqual(np.sum(env.R), 0)

    def test_nonfinite_actions_do_not_advance_episode(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                env = self.make_env(InvManagementBacklogEnv, [0, 10])
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    env.step(np.array([value, 0], dtype=np.float64))
                self.assertEqual(env.period, 0)
                self.assertEqual(np.sum(env.R), 0)


    def test_lost_sales_policy_cannot_be_overridden_by_env_config(self):
        config = {
            "periods": 2, "I0": [0, 0], "r": [1.0] * 3,
            "k": [0.0] * 3, "h": [0.0] * 2,
            "c": [5, 5], "L": [1, 1],
            "dist": 5, "user_D": [4, 3], "backlog": True,
        }
        env = InvManagementLostSalesEnv(env_config=config, backlog=True)
        self.assertTrue(config["backlog"])  # No mutation of caller configuration
        self.assertFalse(env.backlog)
        self.assertTrue(np.all(env.observation_space.low >= 0))
        env.reset(seed=42)
        zero_orders = np.zeros(env.action_space.shape, dtype=np.int64)
        first_obs, _, _, _, first = env.step(zero_orders)
        self.assertEqual(first["unfulfilled"][0], 4)
        self.assertEqual(env.LS[0, 0], 4)
        self.assertEqual(env.B[1, 0], 0)
        self.assertTrue(env.observation_space.contains(first_obs))
        second_obs, _, _, truncated, second = env.step(zero_orders)
        self.assertTrue(truncated)
        self.assertEqual(second["unfulfilled"][0], 3)
        self.assertEqual(env.LS[1, 0], 3)
        self.assertEqual(env.B[2, 0], 0)
        self.assertTrue(env.observation_space.contains(second_obs))

    def test_backlog_policy_cannot_be_overridden_by_env_config(self):
        config = {
            "periods": 2, "I0": [0, 0], "r": [1.0] * 3,
            "k": [0.0] * 3, "h": [0.0] * 2,
            "c": [5, 5], "L": [1, 1],
            "dist": 5, "user_D": [4, 3], "backlog": False,
        }
        env = InvManagementBacklogEnv(env_config=config, backlog=False)
        self.assertFalse(config["backlog"])  # No mutation of caller configuration
        self.assertTrue(env.backlog)
        env.reset(seed=42)
        zero_orders = np.zeros(env.action_space.shape, dtype=np.int64)
        _, _, _, _, first = env.step(zero_orders)
        self.assertEqual(first["unfulfilled"][0], 4)
        self.assertEqual(env.B[1, 0], 4)
        self.assertEqual(env.LS[0, 0], 0)
        _, _, _, truncated, second = env.step(zero_orders)
        self.assertTrue(truncated)
        self.assertEqual(second["unfulfilled"][0], 7)
        self.assertEqual(env.B[2, 0], 7)
        self.assertEqual(env.LS[1, 0], 0)

    def test_policy_defaults_stay_intact_without_env_config(self):
        self.assertTrue(InvManagementBacklogEnv().backlog)
        self.assertFalse(InvManagementLostSalesEnv().backlog)


if __name__ == "__main__":
    unittest.main()
