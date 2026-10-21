"""Regression tests for NewsvendorEnv inventory constraints."""

import unittest

import numpy as np

from newsvendor import NewsvendorEnv


class NewsvendorInventoryCapTests(unittest.TestCase):
    def test_zero_lead_time_caps_immediate_sales(self):
        env = NewsvendorEnv(lead_time=0, max_inventory=3, max_order_quantity=10)
        env.reset(seed=42)
        # Deterministic seeded Poisson demand comfortably exceeds the cap.
        env.mu = 100.0
        env.state[4] = env.mu

        _, _, _, _, info = env.step(np.array([10.0], dtype=np.float32))

        self.assertGreater(info["demand"], 3)
        self.assertAlmostEqual(info["purchase_cost"], 3 * env.cost)
        self.assertAlmostEqual(float(info["revenue"]), 3 * env.price, delta=1e-4)

    def test_zero_inventory_cap_prevents_immediate_sales(self):
        env = NewsvendorEnv(lead_time=0, max_inventory=0, max_order_quantity=10)
        env.reset(seed=42)
        env.mu = 100.0
        env.state[4] = env.mu

        _, _, _, _, info = env.step(np.array([10.0], dtype=np.float32))

        self.assertGreater(info["demand"], 0)
        self.assertEqual(info["revenue"], 0)
        self.assertEqual(info["purchase_cost"], 0)


    def test_positive_lead_time_delays_sales_until_arrival(self):
        env = NewsvendorEnv(lead_time=1, max_inventory=3, max_order_quantity=10)
        env.reset(seed=42)
        env.mu = 100.0
        env.state[4] = env.mu

        _, _, _, _, first = env.step(np.array([10.0], dtype=np.float32))
        self.assertGreater(first["demand"], 3)
        self.assertEqual(first["revenue"], 0)
        self.assertAlmostEqual(first["purchase_cost"], 3 * env.cost)

        _, _, _, _, second = env.step(np.array([0.0], dtype=np.float32))
        self.assertGreater(second["demand"], 3)
        self.assertAlmostEqual(float(second["revenue"]), 3 * env.price, delta=1e-4)
        self.assertEqual(second["purchase_cost"], 0)


    def test_rejects_nonfinite_orders_before_advancing(self):
        for lead_time in (0, 2):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(lead_time=lead_time, value=value):
                    env = NewsvendorEnv(lead_time=lead_time)
                    observation, _ = env.reset(seed=42)
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        env.step(np.array([value], dtype=np.float64))
                    self.assertEqual(env.step_count, 0)
                    np.testing.assert_array_equal(env.state, observation)

    def test_rejects_malformed_action_arrays_before_advancing(self):
        for value in (
            np.array([], dtype=np.float32),
            np.array([1, 2], dtype=np.float32),
            np.array([[1]], dtype=np.float32),
        ):
            with self.subTest(shape=value.shape):
                env = NewsvendorEnv(lead_time=2)
                observation, _ = env.reset(seed=42)
                with self.assertRaisesRegex(ValueError, "Expected a scalar or a one-element action"):
                    env.step(value)
                self.assertEqual(env.step_count, 0)
                np.testing.assert_array_equal(env.state, observation)

    def test_scalar_and_one_element_array_orders_are_equivalent(self):
        for action in (3.0, np.array(3.0), np.array([3.0], dtype=np.float32)):
            with self.subTest(action=repr(action)):
                env = NewsvendorEnv(lead_time=0, max_inventory=10)
                env.reset(seed=42)
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(env.step_count, 1)
                self.assertTrue(np.isfinite(reward))
                self.assertAlmostEqual(info["purchase_cost"], 3 * env.cost)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertFalse(terminated)
                self.assertFalse(truncated)

    def test_negative_and_oversized_orders_are_clipped(self):
        for requested, accepted in ((-100.0, 0.0), (1_000.0, 5.0)):
            with self.subTest(requested=requested):
                env = NewsvendorEnv(
                    lead_time=0, max_order_quantity=5, max_inventory=10
                )
                env.reset(seed=42)
                _, _, _, _, info = env.step(np.array([requested]))
                self.assertAlmostEqual(info["purchase_cost"], accepted * env.cost)


if __name__ == "__main__":
    unittest.main()
